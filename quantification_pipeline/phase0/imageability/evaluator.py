"""
Core imageability evaluator orchestrating the evaluation pipeline.

This module coordinates:
- Model loading and inference
- Data processing loop
- Progress tracking and checkpointing
"""

from typing import Optional
from tqdm import tqdm

from .base_model import BaseImageabilityModel
from .config import EvalConfig, get_model_class
from .data_handler import DataHandler, IdiomEntry, ImageabilityResult


class ImageabilityEvaluator:
    """
    Main evaluator class that orchestrates the imageability evaluation pipeline.
    
    Usage:
        config = EvalConfig(test_mode=True)
        evaluator = ImageabilityEvaluator(config)
        evaluator.run()
    """
    
    def __init__(self, config: EvalConfig):
        """
        Initialize evaluator with configuration.
        
        Args:
            config: Evaluation configuration.
        """
        self.config = config
        self.model: Optional[BaseImageabilityModel] = None
        self.data_handler: Optional[DataHandler] = None
    
    def _init_model(self) -> BaseImageabilityModel:
        """Initialize and load the model."""
        model_class = get_model_class(self.config.model_name)
        model = model_class(
            device=self.config.device,
            torch_dtype=self.config.get_torch_dtype(),
        )
        model.load()
        return model
    
    def _init_data_handler(self) -> DataHandler:
        """Initialize data handler and load data."""
        handler = DataHandler(
            idioms_file=self.config.idioms_file,
            prompt_file=self.config.prompt_file,
            output_file=self.config.output_file,
            checkpoint_file=self.config.checkpoint_file,
        )
        
        # Load data
        handler.load_idioms()
        handler.load_prompt()
        
        # Try to resume from checkpoint
        if self.config.resume_from_checkpoint:
            handler.load_checkpoint()
        
        return handler
    
    def evaluate_single(self, idiom: IdiomEntry) -> ImageabilityResult:
        """
        Evaluate imageability for a single idiom.
        
        Args:
            idiom: The idiom entry to evaluate.
            
        Returns:
            ImageabilityResult with probability values.
        """
        # Format prompt - for imageability, we only use the idiom itself
        # (no definition needed as we judge based on literal wording)
        prompt = self.model.format_prompt(
            idiom=idiom.idiom,
            system_prompt=self.data_handler.prompt,
        )
        
        # Get logits and probabilities
        logit_result = self.model.get_yes_no_logits(prompt)
        
        # Create result
        result = ImageabilityResult(
            idiom_id=idiom.idiom_id,
            idiom=idiom.idiom,
            definition=idiom.definition,
            imageability=logit_result.yes_prob,
            yes_logit=logit_result.yes_logit,
            no_logit=logit_result.no_logit,
        )
        
        return result
    
    def run(self) -> None:
        """
        Run the full evaluation pipeline.
        
        This method:
        1. Initializes model and data
        2. Processes pending idioms
        3. Saves checkpoints periodically
        4. Saves final results
        """
        print("=" * 60)
        print("Imageability Evaluation Pipeline")
        print("=" * 60)
        
        # Initialize
        print("\n[1/4] Initializing model...")
        self.model = self._init_model()
        
        print("\n[2/4] Loading data...")
        self.data_handler = self._init_data_handler()
        
        # Get pending items
        pending = self.data_handler.get_pending_idioms()
        
        # Apply test mode limit
        if self.config.test_mode:
            pending = pending[:self.config.test_n_items]
            print(f"TEST MODE: Processing only {len(pending)} item(s)")
        
        if not pending:
            print("\nNo pending items to process.")
            self._finalize()
            return
        
        # Process idioms
        print(f"\n[3/4] Processing {len(pending)} idioms...")
        
        try:
            for i, idiom in enumerate(tqdm(pending, desc="Evaluating")):
                # Evaluate
                result = self.evaluate_single(idiom)
                
                # Store result
                self.data_handler.add_result(result)
                
                # Debug output for verification
                if self.config.test_mode or i < 3:
                    print(f"\n  [{idiom.idiom_id}] {idiom.idiom}")
                    print(f"      P(yes)={result.imageability:.4f}, "
                          f"yes_logit={result.yes_logit:.3f}, "
                          f"no_logit={result.no_logit:.3f}")
                
                # Periodic checkpoint
                if (i + 1) % self.config.save_interval == 0:
                    self.data_handler.save_checkpoint()
        
        except KeyboardInterrupt:
            print("\n\nInterrupted! Saving checkpoint...")
            self.data_handler.save_checkpoint()
            raise
        
        except Exception as e:
            print(f"\n\nError: {e}")
            print("Saving checkpoint before exit...")
            self.data_handler.save_checkpoint()
            raise
        
        # Finalize
        self._finalize()
    
    def _finalize(self) -> None:
        """Save final results and cleanup."""
        print("\n[4/4] Saving results...")
        
        # Save checkpoint (includes all details)
        self.data_handler.save_checkpoint()
        
        # Save final output (clean format)
        self.data_handler.save_final_results()
        
        # Unload model
        if self.model is not None:
            self.model.unload()
        
        print("\n" + "=" * 60)
        print("Evaluation complete!")
        print(f"Total processed: {self.data_handler.completed_count}")
        print(f"Output: {self.config.output_file}")
        print("=" * 60)
