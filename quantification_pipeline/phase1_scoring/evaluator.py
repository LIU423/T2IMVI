"""
Core evaluator orchestrating the visual element verification pipeline.

This module coordinates:
- Model loading and inference
- Data processing loop with checkpoint support
- Progress tracking and result saving
"""

import logging
from typing import Optional, List
from tqdm import tqdm

from .config import ScoringConfig, get_model_class
from .models.base_model import BaseVerifierModel
from .utils.data_handler import DataHandler, IdiomData, ImageInfo
from .utils.checkpoint import CheckpointManager
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
        ))
        
        if not pending_work:
            print("\nNo pending work to process. All images already completed.")
            self._finalize()
            return
        
        print(f"\n[5/5] Processing {len(pending_work)} pending images...")
        
        try:
            for i, (idiom_data, image_info) in enumerate(tqdm(pending_work, desc="Verifying")):
                self.process_single_image(idiom_data, image_info)
                
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
        print(f"Output directory: {self.config.get_output_dir()}")
        print("=" * 60)
