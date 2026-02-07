"""
Unified Phase0 evaluation script for both imageability and transparency.

This script runs both evaluations and saves combined results to a single JSON file.
Output files are named with model name and timestamp.

Usage:
    # Test mode (1 idiom only)
    python run_phase0.py --test
    
    # Full run
    python run_phase0.py
    
    # Specify model
    python run_phase0.py --model qwen3-0.6b
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

from tqdm import tqdm

# Add project root to path for centralized config import
_THIS_DIR = Path(__file__).parent.resolve()
_PROJECT_ROOT = _THIS_DIR.parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# Import centralized path configuration
from project_config import (
    PROJECT_ROOT,
    INPUT_IRFL_NON_NONE_DIR,
    OUTPUT_PHASE0_DIR,
    PHASE0_IMAGEABILITY_PROMPT,
    PHASE0_TRANSPARENCY_PROMPT,
)


def get_timestamp() -> str:
    """Get formatted timestamp for file naming."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def setup_output_paths(model_name: str, timestamp: str) -> dict:
    """
    Setup output paths with model name and timestamp.
    
    Structure:
        data/output/phase0/
            phase0_{model}_{timestamp}.json          <- Combined results
            checkpoints/
                phase0_{model}_{timestamp}_checkpoint.json
    """
    output_dir = OUTPUT_PHASE0_DIR
    checkpoint_dir = output_dir / "checkpoints"
    
    # Ensure directories exist
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Sanitize model name for filename (replace / with _)
    model_safe = model_name.replace("/", "_").replace("\\", "_")
    
    return {
        "output_file": output_dir / f"phase0_{model_safe}_{timestamp}.json",
        "checkpoint_file": checkpoint_dir / f"phase0_{model_safe}_{timestamp}_checkpoint.json",
    }


class UnifiedPhase0Evaluator:
    """
    Unified evaluator that runs both imageability and transparency,
    saving results to a single JSON file.
    """
    
    def __init__(
        self,
        model_name: str,
        device: str,
        output_file: Path,
        checkpoint_file: Path,
        test_mode: bool = False,
        test_n_items: int = 1,
        save_interval: int = 10,
    ):
        self.model_name = model_name
        self.device = device
        self.output_file = output_file
        self.checkpoint_file = checkpoint_file
        self.test_mode = test_mode
        self.test_n_items = test_n_items
        self.save_interval = save_interval
        
        # Results storage: idiom_id -> {idiom, definition, imageability, transparency}
        self._results: Dict[int, Dict[str, Any]] = {}
        self._idioms: List[Dict[str, Any]] = []
        
        # Model and tokenizer (shared)
        self._model = None
        self._tokenizer = None
        self._yes_token_ids: List[int] = []
        self._no_token_ids: List[int] = []
        
        # Prompts
        self._imageability_prompt: str = ""
        self._transparency_prompt: str = ""
    
    def _load_idioms(self) -> None:
        """Load idioms from JSON file."""
        idioms_file = INPUT_IRFL_NON_NONE_DIR / "unique_idioms.json"
        with open(idioms_file, "r", encoding="utf-8") as f:
            self._idioms = json.load(f)
        print(f"Loaded {len(self._idioms)} idioms")
    
    def _load_prompts(self) -> None:
        """Load prompt templates."""
        imageability_file = PHASE0_IMAGEABILITY_PROMPT
        transparency_file = PHASE0_TRANSPARENCY_PROMPT
        
        with open(imageability_file, "r", encoding="utf-8") as f:
            self._imageability_prompt = f.read().strip()
        
        with open(transparency_file, "r", encoding="utf-8") as f:
            self._transparency_prompt = f.read().strip()
        
        print(f"Loaded prompts: imageability ({len(self._imageability_prompt)} chars), "
              f"transparency ({len(self._transparency_prompt)} chars)")
    
    def _load_checkpoint(self) -> bool:
        """Load checkpoint if exists."""
        if not self.checkpoint_file.exists():
            print("No checkpoint found, starting fresh.")
            return False
        
        try:
            with open(self.checkpoint_file, "r", encoding="utf-8") as f:
                checkpoint_data = json.load(f)
            
            for item in checkpoint_data.get("results", []):
                self._results[item["idiom_id"]] = item
            
            print(f"Loaded checkpoint: {len(self._results)} items completed")
            return True
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Error loading checkpoint: {e}. Starting fresh.")
            return False
    
    def _save_checkpoint(self) -> None:
        """Save current progress to checkpoint file."""
        self.checkpoint_file.parent.mkdir(parents=True, exist_ok=True)
        
        checkpoint_data = {
            "last_updated": datetime.now().isoformat(),
            "model": self.model_name,
            "total_completed": len(self._results),
            "results": list(self._results.values()),
        }
        
        with open(self.checkpoint_file, "w", encoding="utf-8") as f:
            json.dump(checkpoint_data, f, indent=2, ensure_ascii=False)
    
    def _save_final_results(self) -> None:
        """Save final results in output format."""
        self.output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Sort by idiom_id for consistent output
        sorted_results = sorted(
            self._results.values(),
            key=lambda r: r["idiom_id"]
        )
        
        with open(self.output_file, "w", encoding="utf-8") as f:
            json.dump(sorted_results, f, indent=2, ensure_ascii=False)
        
        print(f"Final results saved: {len(sorted_results)} items -> {self.output_file}")
    
    def _load_model(self) -> None:
        """Load the model and tokenizer."""
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
        
        # Get model class from registry
        from quantification_pipeline.phase0.imageability.config import get_model_class
        model_class = get_model_class(self.model_name)
        
        print(f"Loading model: {model_class.MODEL_ID}")
        
        self._tokenizer = AutoTokenizer.from_pretrained(
            model_class.MODEL_ID,
            use_fast=True,
            trust_remote_code=True,
        )
        
        device_map = "auto" if self.device == "auto" else self.device
        self._model = AutoModelForCausalLM.from_pretrained(
            model_class.MODEL_ID,
            torch_dtype=torch.float16,
            device_map=device_map,
            trust_remote_code=True,
        ).eval()
        
        # Cache token IDs for "yes" and "no"
        self._yes_token_ids = self._tokenizer.encode("yes", add_special_tokens=False)
        self._no_token_ids = self._tokenizer.encode("no", add_special_tokens=False)
        
        yes_tokens = [self._tokenizer.decode([tid]) for tid in self._yes_token_ids]
        no_tokens = [self._tokenizer.decode([tid]) for tid in self._no_token_ids]
        print(f"Model loaded.")
        print(f"  'yes' tokenizes to {len(self._yes_token_ids)} token(s): {self._yes_token_ids} -> {yes_tokens}")
        print(f"  'no' tokenizes to {len(self._no_token_ids)} token(s): {self._no_token_ids} -> {no_tokens}")
    
    def _unload_model(self) -> None:
        """Unload model from memory."""
        import torch
        if self._model is not None:
            del self._model
            self._model = None
        if self._tokenizer is not None:
            del self._tokenizer
            self._tokenizer = None
        torch.cuda.empty_cache()
        print("Model unloaded.")
    
    def _format_prompt(self, idiom: str, definition: str, system_prompt: str, include_definition: bool) -> str:
        """Format prompt using chat template."""
        if include_definition:
            user_content = f"Idiom: {idiom}\nMeaning: {definition}"
        else:
            user_content = f"Idiom: {idiom}"
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]
        
        prompt = self._tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        
        return prompt
    
    def _compute_sequence_log_prob(
        self,
        input_ids,
        attention_mask,
        target_token_ids: List[int],
    ) -> float:
        """Compute log probability of generating a specific token sequence."""
        import torch
        
        total_log_prob = 0.0
        current_input_ids = input_ids.clone()
        current_attention_mask = attention_mask.clone() if attention_mask is not None else None
        
        for target_tid in target_token_ids:
            outputs = self._model(
                input_ids=current_input_ids,
                attention_mask=current_attention_mask,
            )
            
            last_logits = outputs.logits[0, -1, :]
            log_probs = torch.nn.functional.log_softmax(last_logits, dim=0)
            token_log_prob = log_probs[target_tid].item()
            total_log_prob += token_log_prob
            
            next_token = torch.tensor([[target_tid]], device=self._model.device)
            current_input_ids = torch.cat([current_input_ids, next_token], dim=1)
            
            if current_attention_mask is not None:
                next_mask = torch.ones((1, 1), device=self._model.device, dtype=current_attention_mask.dtype)
                current_attention_mask = torch.cat([current_attention_mask, next_mask], dim=1)
        
        return total_log_prob
    
    def _compute_yes_prob(self, prompt: str) -> float:
        """Compute P(yes) for a given prompt."""
        import math
        import torch
        
        inputs = self._tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(self._model.device)
        attention_mask = inputs.get("attention_mask", None)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self._model.device)
        
        with torch.no_grad():
            yes_log_prob = self._compute_sequence_log_prob(
                input_ids, attention_mask, self._yes_token_ids
            )
            no_log_prob = self._compute_sequence_log_prob(
                input_ids, attention_mask, self._no_token_ids
            )
        
        # Normalize
        max_log_prob = max(yes_log_prob, no_log_prob)
        yes_exp = math.exp(yes_log_prob - max_log_prob)
        no_exp = math.exp(no_log_prob - max_log_prob)
        total = yes_exp + no_exp
        
        return yes_exp / total
    
    def run(self) -> None:
        """Run the full evaluation pipeline."""
        print("=" * 70)
        print("PHASE 0 UNIFIED EVALUATION")
        print("=" * 70)
        
        # Load data
        print("\n[1/5] Loading data...")
        self._load_idioms()
        self._load_prompts()
        self._load_checkpoint()
        
        # Determine pending items
        completed_ids = set(self._results.keys())
        pending_idioms = [
            idiom for idiom in self._idioms
            if idiom["idiom_id"] not in completed_ids
        ]
        
        # Apply test mode limit
        if self.test_mode:
            pending_idioms = pending_idioms[:self.test_n_items]
            print(f"TEST MODE: Processing only {len(pending_idioms)} item(s)")
        
        if not pending_idioms:
            print("\nNo pending items to process.")
            self._save_final_results()
            return
        
        # Load model
        print("\n[2/5] Loading model...")
        self._load_model()
        
        # Process idioms
        print(f"\n[3/5] Processing {len(pending_idioms)} idioms...")
        
        try:
            for i, idiom in enumerate(tqdm(pending_idioms, desc="Evaluating")):
                idiom_id = idiom["idiom_id"]
                idiom_text = idiom["idiom"]
                definition = idiom["definition"]
                
                # Compute imageability (no definition needed)
                imageability_prompt = self._format_prompt(
                    idiom_text, definition, self._imageability_prompt, include_definition=False
                )
                imageability = self._compute_yes_prob(imageability_prompt)
                
                # Compute transparency (with definition)
                transparency_prompt = self._format_prompt(
                    idiom_text, definition, self._transparency_prompt, include_definition=True
                )
                transparency = self._compute_yes_prob(transparency_prompt)
                
                # Store result
                self._results[idiom_id] = {
                    "idiom_id": idiom_id,
                    "idiom": idiom_text,
                    "definition": definition,
                    "imageability": round(imageability, 6),
                    "transparency": round(transparency, 6),
                }
                
                # Debug output
                if self.test_mode or i < 3:
                    print(f"\n  [{idiom_id}] {idiom_text}")
                    print(f"      imageability={imageability:.4f}, transparency={transparency:.4f}")
                
                # Periodic checkpoint
                if (i + 1) % self.save_interval == 0:
                    self._save_checkpoint()
        
        except KeyboardInterrupt:
            print("\n\nInterrupted! Saving checkpoint...")
            self._save_checkpoint()
            raise
        
        except Exception as e:
            print(f"\n\nError: {e}")
            print("Saving checkpoint before exit...")
            self._save_checkpoint()
            raise
        
        # Finalize
        print("\n[4/5] Saving results...")
        self._save_checkpoint()
        self._save_final_results()
        
        print("\n[5/5] Cleanup...")
        self._unload_model()
        
        print("\n" + "=" * 70)
        print("Evaluation complete!")
        print(f"Total processed: {len(self._results)}")
        print(f"Output: {self.output_file}")
        print("=" * 70)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run Phase0 evaluation (imageability + transparency) -> single JSON output.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument(
        "--test",
        action="store_true",
        help="Test mode: process only 1 idiom for verification.",
    )
    
    parser.add_argument(
        "--test-n",
        type=int,
        default=1,
        help="Number of items to process in test mode (default: 1).",
    )
    
    parser.add_argument(
        "--model",
        type=str,
        default="qwen3-0.6b",
        help="Model to use (default: qwen3-0.6b).",
    )
    
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu", "auto"],
        help="Device to run model on (default: cuda). Use 'auto' for multi-GPU offload.",
    )
    
    parser.add_argument(
        "--save-interval",
        type=int,
        default=10,
        help="Save checkpoint every N items (default: 10).",
    )
    
    return parser.parse_args()


def main() -> int:
    """Main entry point."""
    args = parse_args()
    
    # Generate timestamp for this run
    timestamp = get_timestamp()
    
    # Setup paths
    paths = setup_output_paths(args.model, timestamp)
    
    print(f"Configuration:")
    print(f"  Model: {args.model}")
    print(f"  Device: {args.device}")
    print(f"  Test mode: {args.test}")
    if args.test:
        print(f"  Test items: {args.test_n}")
    print(f"  Timestamp: {timestamp}")
    print(f"  Output: {paths['output_file']}")
    print(f"  Checkpoint: {paths['checkpoint_file']}")
    
    try:
        evaluator = UnifiedPhase0Evaluator(
            model_name=args.model,
            device=args.device,
            output_file=paths["output_file"],
            checkpoint_file=paths["checkpoint_file"],
            test_mode=args.test,
            test_n_items=args.test_n,
            save_interval=args.save_interval,
        )
        evaluator.run()
        return 0
    except KeyboardInterrupt:
        print("\nAborted by user.")
        return 130
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
