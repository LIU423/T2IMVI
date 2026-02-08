"""
Main evaluator/orchestrator for Phase 2 AEA calculation pipeline.

This module coordinates the entire AEA calculation workflow:
1. Initialize model, data handler, checkpoint manager
2. Discover images to process
3. Process images in batches with checkpoint support
4. Save AEA scores to output files
"""

import logging
import json
import time
import sys
from pathlib import Path
from typing import List, Optional
from datetime import datetime

from tqdm import tqdm

from .config import AEAConfig, get_model_class
from .models.base_model import BaseAEAModel
from .calculators.aea_calculator import AEACalculator
from .utils.checkpoint import CheckpointManager
from .utils.data_handler import DataHandler, ImageInfo

logger = logging.getLogger(__name__)


class AEAEvaluator:
    """
    Main orchestrator for AEA calculation pipeline.
    
    Coordinates model loading, data discovery, processing loop,
    checkpointing, and result saving.
    """
    
    def __init__(self, config: AEAConfig):
        """
        Initialize evaluator with configuration.
        
        Args:
            config: AEAConfig with all settings
        """
        self.config = config
        self.model: Optional[BaseAEAModel] = None
        self.data_handler: Optional[DataHandler] = None
        self.checkpoint_manager: Optional[CheckpointManager] = None
        self.calculator: Optional[AEACalculator] = None
        self._images_since_cleanup: int = 0
        self._timing_total_seconds: float = 0.0
        self._timing_attempted_images: int = 0
        self._timing_success_images: int = 0

    def _cleanup_after_images(self) -> None:
        """Best-effort CUDA memory cleanup."""
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
    
    def _init_model(self) -> BaseAEAModel:
        """Initialize and load the VLM model."""
        model_class = get_model_class(self.config.model_key)
        model = model_class(
            model_id=self.config.model_id,
            device=self.config.device,
        )
        model.load()
        return model
    
    def _init_data_handler(self) -> DataHandler:
        """Initialize data handler."""
        return DataHandler(
            phase1_output_dir=self.config.phase1_output_dir,
            images_root=self.config.images_root,
            prompt_file=self.config.prompt_file,
        )
    
    def _init_checkpoint_manager(self) -> CheckpointManager:
        """Initialize checkpoint manager."""
        return CheckpointManager(
            checkpoint_path=self.config.checkpoint_file,
            model_name=self.config.model_key,
        )
    
    def _discover_pending_images(self) -> List[ImageInfo]:
        """
        Discover images that need processing.
        
        Filters out already-completed images based on checkpoint.
        
        Returns:
            List of ImageInfo for pending images
        """
        # Discover all available images
        max_per_idiom = self.config.test_n_images if self.config.test_mode else None
        
        all_images = self.data_handler.discover_all_images(
            idiom_ids=self.config.idiom_ids,
            max_images_per_idiom=max_per_idiom,
        )
        
        if not self.config.resume:
            return all_images
        
        # Filter out completed images
        pending = []
        for img_info in all_images:
            if not self.checkpoint_manager.is_completed(
                img_info.idiom_id, img_info.image_id
            ):
                pending.append(img_info)
            else:
                logger.debug(f"Skipping completed: {img_info.key}")
        
        logger.info(
            f"Pending: {len(pending)} images "
            f"(already completed: {len(all_images) - len(pending)})"
        )
        
        return pending
    
    def process_single_image(self, image_info: ImageInfo) -> float:
        """
        Process a single image and return AEA score.
        
        Args:
            image_info: ImageInfo for the image
            
        Returns:
            Computed AEA score
        """
        # Load figurative data
        figurative_data = self.data_handler.load_figurative_data(image_info)
        
        # Load image
        image = self.data_handler.load_image(image_info)
        
        # Calculate AEA score
        aea_score = self.calculator.calculate_for_image_info(
            image_info=image_info,
            figurative_data=figurative_data,
            image=image,
        )
        
        # Save result
        self.data_handler.save_aea_score(image_info, aea_score)
        
        return aea_score
    
    def run(self) -> None:
        """
        Run the full AEA calculation pipeline.
        
        Main entry point for the evaluator.
        """
        print("\n" + "=" * 60)
        print("Phase 2 Calculation: AEA (Abstract Element Alignment)")
        print("=" * 60)
        
        try:
            # Step 1: Initialize model
            print("\n[1/5] Initializing model...")
            self.model = self._init_model()
            
            # Step 2: Initialize data handler
            print("\n[2/5] Initializing data handler...")
            self.data_handler = self._init_data_handler()
            
            # Step 3: Initialize checkpoint manager
            print("\n[3/5] Loading checkpoint...")
            self.checkpoint_manager = self._init_checkpoint_manager()
            print(f"  Previously completed: {self.checkpoint_manager.get_completed_count()} images")
            
            # Step 4: Initialize calculator
            print("\n[4/5] Initializing AEA calculator...")
            print(f"  Score threshold: {self.config.score_threshold} (entities/actions below this → AEA=0)")
            self.calculator = AEACalculator(
                model=self.model,
                prompt_template=self.data_handler.prompt_template,
                score_threshold=self.config.score_threshold,
            )
            
            # Test mode message
            if self.config.test_mode:
                idiom_count = len(self.config.idiom_ids) if self.config.idiom_ids else "all"
                print(f"\nTEST MODE: Processing {idiom_count} idiom(s), "
                      f"max {self.config.test_n_images} images each")
            
            # Discover pending images
            pending_images = self._discover_pending_images()
            
            if not pending_images:
                print("\nNo pending images to process. All done!")
                return
            
            print(f"\n[5/5] Processing {len(pending_images)} pending images...")
            print()
            
            # Process images
            processed_count = 0
            with tqdm(pending_images, desc="Calculating AEA") as pbar:
                for image_info in pbar:
                    pbar.set_postfix_str(f"{image_info.key}")
                    
                    try:
                        t0 = time.perf_counter()
                        aea_score = self.process_single_image(image_info)
                        elapsed = time.perf_counter() - t0
                        
                        # Update checkpoint
                        self.checkpoint_manager.mark_completed(
                            image_info.idiom_id, image_info.image_id
                        )
                        processed_count += 1
                        self._timing_total_seconds += elapsed
                        self._timing_attempted_images += 1
                        self._timing_success_images += 1
                        
                        # Save checkpoint periodically
                        if processed_count % self.config.checkpoint_interval == 0:
                            self.checkpoint_manager.save()
                        
                        logger.info(
                            f"Processed {image_info.key}: AEA={aea_score:.4f}"
                        )
                        
                    except Exception as e:
                        self._timing_attempted_images += 1
                        logger.error(f"Error processing {image_info.key}: {e}")
                        raise
                    finally:
                        self._images_since_cleanup += 1
                        if self._images_since_cleanup >= 10:
                            self._cleanup_after_images()
                            self._images_since_cleanup = 0
            
            # Final checkpoint save
            self.checkpoint_manager.save()
            
            print("\n" + "-" * 60)
            print("Finalizing...")
            
        finally:
            # Cleanup
            if self.model is not None:
                self.model.unload()
        
        print(f"\n{'=' * 60}")
        print("AEA calculation complete!")
        print(f"Total processed: {processed_count} images")
        self._print_and_save_timing_summary()
        print(f"Output directory: {self.config.phase1_output_dir}")
        print("=" * 60 + "\n")

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
            "phase": "phase2_aea",
            "model_key": self.config.model_key,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "attempted_images": attempted,
            "processed_success_images": success,
            "failed_images": attempted - success,
            "total_seconds": self._timing_total_seconds,
            "avg_seconds_per_attempted_image": avg_attempted,
            "avg_seconds_per_success_image": avg_success,
        }
        out_path = self.config.phase1_output_dir / "timing_phase2_aea.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
