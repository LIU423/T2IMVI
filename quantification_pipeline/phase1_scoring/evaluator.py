"""
Core evaluator orchestrating the visual element verification pipeline.

This module coordinates:
- Model loading and inference
- Data processing loop with checkpoint support
- Progress tracking and result saving
"""

import logging
import time
import json
from datetime import datetime
from typing import Optional, List
from tqdm import tqdm

from .config import ScoringConfig, get_model_class
from .models.base_model import BaseVerifierModel
from .utils.data_handler import DataHandler, IdiomData, ImageInfo
from .utils.checkpoint import CheckpointManager
from .utils.failed_items import FailedItemLogger
from .verifiers.figurative_verifier import FigurativeVerifier
from .verifiers.literal_verifier import LiteralVerifier

logger = logging.getLogger(__name__)


class ScoringEvaluator:
    """
    Main evaluator class that orchestrates the visual element verification pipeline.
    
    Features:
    - Checkpoint/resume support for long-running jobs
    - Processes all entities and actions for both figurative and literal tracks
    - Saves results in structured JSON format
    
    Usage:
        config = ScoringConfig(test_mode=True)
        evaluator = ScoringEvaluator(config)
        evaluator.run()
    """
    
    def __init__(self, config: ScoringConfig):
        """
        Initialize evaluator with configuration.
        
        Args:
            config: Scoring configuration.
        """
        self.config = config
        self.model: Optional[BaseVerifierModel] = None
        self.data_handler: Optional[DataHandler] = None
        self.checkpoint: Optional[CheckpointManager] = None
        self.figurative_verifier: Optional[FigurativeVerifier] = None
        self.literal_verifier: Optional[LiteralVerifier] = None
        self.failed_items_logger: Optional[FailedItemLogger] = None
        self.failed_count: int = 0
        self._images_since_cleanup: int = 0
        self._timing_total_seconds: float = 0.0
        self._timing_attempted_images: int = 0
        self._timing_success_images: int = 0
    
    def _init_model(self) -> BaseVerifierModel:
        """Initialize and load the VLM model."""
        model_class = get_model_class(self.config.model_name)
        model = model_class(
            device=self.config.device,
            torch_dtype=self.config.get_torch_dtype(),
        )
        model.load()
        return model
    
    def _init_data_handler(self) -> DataHandler:
        """Initialize data handler."""
        return DataHandler(
            input_images_dir=self.config.input_images_dir,
            extraction_output_dir=self.config.extraction_output_dir,
            figurative_prompt_file=self.config.figurative_prompt_file,
            literal_prompt_file=self.config.literal_prompt_file,
            output_dir=self.config.get_output_dir(),
        )
    
    def _init_checkpoint(self) -> CheckpointManager:
        """Initialize checkpoint manager."""
        checkpoint = CheckpointManager(
            checkpoint_file=self.config.get_checkpoint_file(),
            model_name=self.config.model_name,
        )
        
        if self.config.resume_from_checkpoint:
            checkpoint.load()
        
        return checkpoint
    
    def _init_verifiers(self) -> None:
        """Initialize figurative and literal verifiers."""
        self.data_handler.load_prompts()
        
        self.figurative_verifier = FigurativeVerifier(
            model=self.model,
            prompt_template=self.data_handler.figurative_prompt,
        )
        
        self.literal_verifier = LiteralVerifier(
            model=self.model,
            prompt_template=self.data_handler.literal_prompt,
        )

    def _init_failed_items_logger(self) -> FailedItemLogger:
        """Initialize failed-items JSONL logger."""
        return FailedItemLogger(self.config.get_failed_items_file())

    def _is_oom_error(self, error: Exception) -> bool:
        """Heuristic check for CUDA/VRAM OOM errors."""
        msg = str(error).lower()
        return (
            "out of memory" in msg
            or "cuda out of memory" in msg
            or "cuda error: out of memory" in msg
            or "cudnn_status_alloc_failed" in msg
        )

    def _cleanup_after_oom(self) -> None:
        """Try to release CUDA memory before retry."""
        try:
            import gc
            import torch

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                if hasattr(torch.cuda, "ipc_collect"):
                    torch.cuda.ipc_collect()
        except Exception:
            # Best-effort cleanup only.
            pass

    def _cleanup_after_image(self) -> None:
        """Best-effort cleanup after each image to cap VRAM growth."""
        try:
            import gc
            import torch

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                if hasattr(torch.cuda, "ipc_collect"):
                    torch.cuda.ipc_collect()
        except Exception:
            # Best-effort cleanup only.
            pass

    def _process_with_retry(self, idiom_data: IdiomData, image_info: ImageInfo) -> bool:
        """Process one image with OOM-only retries and failed-item logging."""
        try:
            max_attempts = max(1, int(self.config.max_oom_attempts))
            last_error: Optional[Exception] = None

            for attempt in range(1, max_attempts + 1):
                try:
                    self.process_single_image(idiom_data, image_info)
                    return True
                except Exception as exc:
                    last_error = exc
                    is_oom = self._is_oom_error(exc)
                    can_retry = is_oom and attempt < max_attempts
                    if can_retry:
                        logger.warning(
                            "OOM on idiom_%s/%s (attempt %s/%s). Retrying...",
                            idiom_data.idiom_id,
                            image_info.image_id,
                            attempt,
                            max_attempts,
                        )
                        self._cleanup_after_oom()
                        if self.config.oom_retry_backoff_seconds > 0:
                            time.sleep(self.config.oom_retry_backoff_seconds)
                        continue
                    break

            assert last_error is not None
            self.failed_count += 1
            is_oom = self._is_oom_error(last_error)
            if self.failed_items_logger is not None:
                record = self.failed_items_logger.make_record(
                    model_name=self.config.model_name,
                    idiom_id=idiom_data.idiom_id,
                    image_num=image_info.image_num,
                    image_id=image_info.image_id,
                    is_oom=is_oom,
                    attempts=max_attempts if is_oom else 1,
                    error=last_error,
                )
                self.failed_items_logger.append(record)

            logger.error(
                "Failed idiom_%s/%s after %s attempt(s): %s",
                idiom_data.idiom_id,
                image_info.image_id,
                max_attempts if is_oom else 1,
                last_error,
            )
            if not self.config.continue_on_error:
                raise last_error
            return False
        finally:
            self._images_since_cleanup += 1
            if self._images_since_cleanup >= 10:
                self._cleanup_after_image()
                self._images_since_cleanup = 0
    
    def process_single_image(
        self,
        idiom_data: IdiomData,
        image_info: ImageInfo,
    ) -> None:
        """
        Process a single idiom-image pair.
        
        Verifies all figurative and literal elements and saves results.
        
        Args:
            idiom_data: The idiom extraction data
            image_info: The image to process
        """
        logger.info(f"Processing idiom_{idiom_data.idiom_id}/{image_info.image_id}")
        
        # Load image
        image = self.data_handler.load_image(image_info)
        
        # Verify figurative elements
        logger.info("  Figurative verification:")
        figurative_scores = self.figurative_verifier.verify_all_elements(
            image=image,
            entities=idiom_data.figurative_entities,
            actions=idiom_data.figurative_actions,
        )
        
        # Verify literal elements
        logger.info("  Literal verification:")
        literal_scores = self.literal_verifier.verify_all_elements(
            image=image,
            entities=idiom_data.literal_entities,
            actions=idiom_data.literal_actions,
        )
        
        # Save results
        self.data_handler.save_verification_results(
            idiom_id=idiom_data.idiom_id,
            image_num=image_info.image_num,
            figurative_results=figurative_scores,
            literal_results=literal_scores,
            idiom_data=idiom_data,
        )
        
        # Update checkpoint
        self.checkpoint.mark_completed(
            idiom_id=f"idiom_{idiom_data.idiom_id}",
            image_id=image_info.image_id,
        )
    
    def run(self) -> None:
        """
        Run the full verification pipeline.
        
        This method:
        1. Initializes model, data handler, and checkpoint
        2. Iterates over pending idiom-image pairs
        3. Verifies all elements and saves results
        4. Supports resume from checkpoint on interruption
        """
        print("=" * 60)
        print("Phase 1 Scoring: Visual Element Verification Pipeline")
        print("=" * 60)
        
        # Initialize components
        print("\n[1/5] Initializing model...")
        self.model = self._init_model()
        
        print("\n[2/5] Initializing data handler...")
        self.data_handler = self._init_data_handler()
        
        print("\n[3/5] Loading checkpoint...")
        self.checkpoint = self._init_checkpoint()
        print(f"  Previously completed: {self.checkpoint.get_completed_count()} images")

        self.failed_items_logger = self._init_failed_items_logger()
        print(f"  Failed items log: {self.config.get_failed_items_file()}")
        
        print("\n[4/5] Initializing verifiers...")
        self._init_verifiers()
        
        # Determine idiom IDs to process
        idiom_ids = self.config.idiom_ids
        if idiom_ids is None:
            idiom_ids = self.data_handler.get_available_idiom_ids()
        
        # Apply test mode limits
        max_images = None
        if self.config.test_mode:
            idiom_ids = idiom_ids[:self.config.test_n_idioms]
            max_images = self.config.test_n_images
            print(f"\nTEST MODE: Processing {len(idiom_ids)} idiom(s), max {max_images} images each")
        
        # Collect pending work
        pending_work = list(self.data_handler.iter_pending_work(
            idiom_ids=idiom_ids,
            max_images_per_idiom=max_images,
            target_image_nums_by_idiom=self.config.target_image_nums_by_idiom,
        ))
        
        if not pending_work:
            print("\nNo pending work to process. All images already completed.")
            self._finalize()
            return
        
        print(f"\n[5/5] Processing {len(pending_work)} pending images...")
        
        try:
            for i, (idiom_data, image_info) in enumerate(tqdm(pending_work, desc="Verifying")):
                t0 = time.perf_counter()
                ok = self._process_with_retry(idiom_data, image_info)
                elapsed = time.perf_counter() - t0
                self._timing_total_seconds += elapsed
                self._timing_attempted_images += 1
                if ok:
                    self._timing_success_images += 1
                
                # Periodic checkpoint save
                if (i + 1) % self.config.save_interval == 0:
                    self.checkpoint.save()
        
        except KeyboardInterrupt:
            print("\n\nInterrupted! Saving checkpoint...")
            self.checkpoint.save()
            raise
        
        except Exception as e:
            print(f"\n\nError: {e}")
            print("Saving checkpoint before exit...")
            self.checkpoint.save()
            raise
        
        # Finalize
        self._finalize()
    
    def _finalize(self) -> None:
        """Save final checkpoint and cleanup."""
        print("\n" + "-" * 60)
        print("Finalizing...")
        
        # Save final checkpoint
        if self.checkpoint is not None:
            self.checkpoint.save()
        
        # Unload model
        if self.model is not None:
            self.model.unload()
        
        print("\n" + "=" * 60)
        print("Verification complete!")
        if self.checkpoint is not None:
            print(f"Total processed: {self.checkpoint.get_completed_count()} images")
        print(f"Failed images: {self.failed_count}")
        self._print_and_save_timing_summary()
        print(f"Failed log file: {self.config.get_failed_items_file()}")
        print(f"Output directory: {self.config.get_output_dir()}")
        print("=" * 60)

    def _print_and_save_timing_summary(self) -> None:
        attempted = self._timing_attempted_images
        success = self._timing_success_images
        avg_attempted = (self._timing_total_seconds / attempted) if attempted else 0.0
        avg_success = (self._timing_total_seconds / success) if success else 0.0
        print(
            "Timing summary: "
            f"total={self._timing_total_seconds:.2f}s, "
            f"avg/attempted_image={avg_attempted:.3f}s, "
            f"avg/success_image={avg_success:.3f}s"
        )
        summary = {
            "phase": "phase1_scoring",
            "model_name": self.config.model_name,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "attempted_images": attempted,
            "processed_success_images": success,
            "failed_images": self.failed_count,
            "total_seconds": self._timing_total_seconds,
            "avg_seconds_per_attempted_image": avg_attempted,
            "avg_seconds_per_success_image": avg_success,
        }
        out_path = self.config.get_output_dir() / "timing_phase1_scoring.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
