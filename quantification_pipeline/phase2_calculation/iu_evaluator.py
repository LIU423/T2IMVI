"""
Main evaluator/orchestrator for Phase 2 IU (Image Understanding) calculation pipeline.

This module coordinates the entire IU calculation workflow:
1. Initialize model, data handler, checkpoint manager
2. Discover images to process
3. Process images with checkpoint support
4. Save IU scores to output files
"""

import logging
import json
import time
import sys
from pathlib import Path
from typing import List, Optional
from datetime import datetime

from tqdm import tqdm

from .config import IUConfig, MODEL_REGISTRY
from .models.iu_base_model import BaseIUModel
from .models.qwen_vl_iu_model import (
    Qwen3VLIUModel,
    Qwen3VL30BA3BInstructIUModel,
)
from .calculators.iu_calculator import IUCalculator
from .utils.checkpoint import CheckpointManager
from .utils.data_handler import DataHandler, ImageInfo

logger = logging.getLogger(__name__)


# Model registry for IU (maps model_key to class)
IU_MODEL_REGISTRY = {
    "qwen3-vl-2b": Qwen3VLIUModel,
    "qwen3-vl-30b-a3b-instruct": Qwen3VL30BA3BInstructIUModel,
}


def get_iu_model_class(model_key: str):
    """
    Get the IU model class for a model key.
    
    Args:
        model_key: Key in MODEL_REGISTRY
        
    Returns:
        The IU model class
    """
    if model_key not in IU_MODEL_REGISTRY:
        raise ValueError(
            f"Unknown IU model: {model_key}. "
            f"Available: {list(IU_MODEL_REGISTRY.keys())}"
        )
    return IU_MODEL_REGISTRY[model_key]


class IUDataHandler(DataHandler):
    """
    Extended data handler for IU pipeline.
    
    Adds methods for:
    - Loading IU prompt templates
    - Saving IU scores to figurative_score.json
    """
    
    def __init__(
        self,
        phase1_output_dir: Path,
        images_root: Path,
        relationships_prompt_file: Path,
        without_relationships_prompt_file: Path,
    ):
        """
        Initialize IU data handler.
        
        Args:
            phase1_output_dir: Directory containing Phase 1 output
            images_root: Root directory for images
            relationships_prompt_file: Path to phase2_iu_relationships.txt
            without_relationships_prompt_file: Path to phase2_iu_without_relationships.txt
        """
        # Use a dummy prompt file for parent class (we override prompt loading)
        super().__init__(
            phase1_output_dir=phase1_output_dir,
            images_root=images_root,
            prompt_file=relationships_prompt_file,  # Not used directly
        )
        self.relationships_prompt_file = relationships_prompt_file
        self.without_relationships_prompt_file = without_relationships_prompt_file
        self._relationships_prompt_template: Optional[str] = None
        self._without_relationships_prompt_template: Optional[str] = None
    
    @property
    def relationships_prompt_template(self) -> str:
        """Load and cache the relationships prompt template."""
        if self._relationships_prompt_template is None:
            with open(self.relationships_prompt_file, 'r', encoding='utf-8') as f:
                self._relationships_prompt_template = f.read()
            logger.info(f"Loaded relationships prompt from {self.relationships_prompt_file}")
        return self._relationships_prompt_template
    
    @property
    def without_relationships_prompt_template(self) -> str:
        """Load and cache the without-relationships prompt template."""
        if self._without_relationships_prompt_template is None:
            with open(self.without_relationships_prompt_file, 'r', encoding='utf-8') as f:
                self._without_relationships_prompt_template = f.read()
            logger.info(
                f"Loaded without-relationships prompt from {self.without_relationships_prompt_file}"
            )
        return self._without_relationships_prompt_template
    
    def save_iu_score(
        self,
        image_info: ImageInfo,
        iu_score: float,
    ) -> None:
        """
        Save IU score to figurative_score.json.
        
        If the file already exists, adds/updates the 'iu' key.
        If not, creates a new file with the 'iu' key.
        
        Args:
            image_info: ImageInfo with output path
            iu_score: Computed IU score
        """
        import json
        
        output_path = image_info.figurative_score_json_path
        
        # Load existing data if file exists
        if output_path.exists():
            with open(output_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        else:
            data = {}
        
        # Add/update IU score
        data["iu"] = iu_score
        
        # Ensure directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
        
        logger.debug(f"Saved IU score {iu_score:.4f} to {output_path}")


class IUEvaluator:
    """
    Main orchestrator for IU calculation pipeline.
    
    Coordinates model loading, data discovery, processing loop,
    checkpointing, and result saving.
    """
    
    def __init__(self, config: IUConfig):
        """
        Initialize evaluator with configuration.
        
        Args:
            config: IUConfig with all settings
        """
        self.config = config
        self.model: Optional[BaseIUModel] = None
        self.data_handler: Optional[IUDataHandler] = None
        self.checkpoint_manager: Optional[CheckpointManager] = None
        self.calculator: Optional[IUCalculator] = None
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
    
    def _init_model(self) -> BaseIUModel:
        """Initialize and load the VLM model."""
        model_class = get_iu_model_class(self.config.model_key)
        model = model_class(
            model_id=self.config.model_id,
            device=self.config.device,
        )
        model.load()
        return model
    
    def _init_data_handler(self) -> IUDataHandler:
        """Initialize data handler."""
        return IUDataHandler(
            phase1_output_dir=self.config.phase1_output_dir,
            images_root=self.config.images_root,
            relationships_prompt_file=self.config.relationships_prompt_file,
            without_relationships_prompt_file=self.config.without_relationships_prompt_file,
        )
    
    def _init_checkpoint_manager(self) -> CheckpointManager:
        """Initialize checkpoint manager."""
        return CheckpointManager(
            checkpoint_path=self.config.checkpoint_file,
            model_name=f"{self.config.model_key}_iu",
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
        Process a single image and return IU score.
        
        Args:
            image_info: ImageInfo for the image
            
        Returns:
            Computed IU score
        """
        # Load figurative data
        figurative_data = self.data_handler.load_figurative_data(image_info)
        
        # Load image
        image = self.data_handler.load_image(image_info)
        
        # Calculate IU score
        iu_score = self.calculator.calculate_for_image_info(
            image_info=image_info,
            figurative_data=figurative_data,
            image=image,
        )
        
        # Save result
        self.data_handler.save_iu_score(image_info, iu_score)
        
        return iu_score
    
    def run(self) -> None:
        """
        Run the full IU calculation pipeline.
        
        Main entry point for the evaluator.
        """
        print("\n" + "=" * 60)
        print("Phase 2 Calculation: IU (Image Understanding)")
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
            print("\n[4/5] Initializing IU calculator...")
            self.calculator = IUCalculator(
                model=self.model,
                relationships_prompt_template=self.data_handler.relationships_prompt_template,
                without_relationships_prompt_template=self.data_handler.without_relationships_prompt_template,
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
            with tqdm(pending_images, desc="Calculating IU") as pbar:
                for image_info in pbar:
                    pbar.set_postfix_str(f"{image_info.key}")
                    
                    try:
                        t0 = time.perf_counter()
                        iu_score = self.process_single_image(image_info)
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
                            f"Processed {image_info.key}: IU={iu_score:.4f}"
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
        print("IU calculation complete!")
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
            "phase": "phase2_iu",
            "model_key": self.config.model_key,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "attempted_images": attempted,
            "processed_success_images": success,
            "failed_images": attempted - success,
            "total_seconds": self._timing_total_seconds,
            "avg_seconds_per_attempted_image": avg_attempted,
            "avg_seconds_per_success_image": avg_success,
        }
        out_path = self.config.phase1_output_dir / "timing_phase2_iu.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
