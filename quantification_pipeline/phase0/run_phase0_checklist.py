"""
Phase0 Checklist-Based Evaluation for Imageability and Transparency.

This script implements a multi-question checklist approach:
- Imageability: 14 yes/no questions, averaged
- Transparency: 15 yes/no questions, averaged

Usage:
    # Test mode (1 idiom only)
    python run_phase0_checklist.py --test
    
    # Full run
    python run_phase0_checklist.py
    
    # Specify model
    python run_phase0_checklist.py --model qwen3-0.6b
"""

import argparse
import json
import math
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Add project root to path
_THIS_DIR = Path(__file__).parent.resolve()
_PROJECT_ROOT = _THIS_DIR.parent.parent  # phase0 -> quantification_pipeline -> T2IMVI
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from project_config import (
    INPUT_IRFL_NON_NONE_DIR,
    OUTPUT_PHASE0_DIR,
    PROMPT_DIR,
)

# Add prompt directory to path for checklist imports
if str(PROMPT_DIR) not in sys.path:
    sys.path.insert(0, str(PROMPT_DIR))

from phase0_imageability_checklist import (
    IMAGEABILITY_SYSTEM_PROMPT,
    IMAGEABILITY_QUESTIONS,
    get_imageability_prompts,
)
from phase0_transparency_checklist import (
    TRANSPARENCY_SYSTEM_PROMPT,
    TRANSPARENCY_QUESTIONS,
    get_transparency_prompts,
)


def get_timestamp() -> str:
    """Get formatted timestamp for file naming."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def setup_output_paths(model_name: str, timestamp: str) -> dict:
    """Setup output paths with model name and timestamp."""
    output_dir = OUTPUT_PHASE0_DIR
    checkpoint_dir = output_dir / "checkpoints"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    model_safe = model_name.replace("/", "_").replace("\\", "_")
    
    return {
        "output_file": output_dir / f"phase0_checklist_{model_safe}_{timestamp}.json",
        "checkpoint_file": checkpoint_dir / f"phase0_checklist_{model_safe}_{timestamp}_checkpoint.json",
    }


class ChecklistEvaluator:
    """
    Evaluator using multi-question checklists for imageability and transparency.
    
    Scoring method:
    1. For each idiom, ask 14 (imageability) or 15 (transparency) yes/no questions
    2. For each question, compute P(yes) / (P(yes) + P(no))
    3. Final score = arithmetic mean of all question scores
    """
    
    # Model registry
    MODEL_REGISTRY = {
        "qwen3-0.6b": "Qwen/Qwen3-0.6B",
        "qwen3-30b-a3b-instruct-2507": "Qwen/Qwen3-30B-A3B-Instruct-2507",
        "Qwen/Qwen3-30B-A3B-Instruct-2507": "Qwen/Qwen3-30B-A3B-Instruct-2507",
    }
    
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
        self.model_id = self.MODEL_REGISTRY.get(model_name, model_name)
        self.device = device
        self.output_file = output_file
        self.checkpoint_file = checkpoint_file
        self.test_mode = test_mode
        self.test_n_items = test_n_items
        self.save_interval = save_interval
        
        # Results storage
        self._results: Dict[int, Dict[str, Any]] = {}
        self._idioms: List[Dict[str, Any]] = []
        
        # Model and tokenizer
        self._model = None
        self._tokenizer = None
        self._yes_token_ids: List[int] = []
        self._no_token_ids: List[int] = []
    
    def _load_idioms(self) -> None:
        """Load idioms from JSON file."""
        idioms_file = INPUT_IRFL_NON_NONE_DIR / "unique_idioms.json"
        with open(idioms_file, "r", encoding="utf-8") as f:
            self._idioms = json.load(f)
        print(f"Loaded {len(self._idioms)} idioms")
    
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
        
        sorted_results = sorted(
            self._results.values(),
            key=lambda r: r["idiom_id"]
        )
        
        with open(self.output_file, "w", encoding="utf-8") as f:
            json.dump(sorted_results, f, indent=2, ensure_ascii=False)
        
        print(f"Final results saved: {len(sorted_results)} items -> {self.output_file}")
    
    def _load_model(self) -> None:
        """Load the model and tokenizer."""
        print(f"Loading model: {self.model_id}")
        
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_id,
            use_fast=True,
            trust_remote_code=True,
        )
        
        device_map = "auto" if self.device == "auto" else self.device
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
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
        if self._model is not None:
            del self._model
            self._model = None
        if self._tokenizer is not None:
            del self._tokenizer
            self._tokenizer = None
        torch.cuda.empty_cache()
        print("Model unloaded.")
    
    def _format_chat_prompt(self, system_prompt: str, user_content: str) -> str:
        """Format prompt using chat template."""
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
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        target_token_ids: List[int],
    ) -> float:
        """Compute log probability of generating a specific token sequence."""
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
    
    @torch.no_grad()
    def _compute_yes_prob(self, prompt: str) -> float:
        """Compute P(yes) / (P(yes) + P(no)) for a given prompt."""
        inputs = self._tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(self._model.device)
        attention_mask = inputs.get("attention_mask", None)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self._model.device)
        
        yes_log_prob = self._compute_sequence_log_prob(
            input_ids, attention_mask, self._yes_token_ids
        )
        no_log_prob = self._compute_sequence_log_prob(
            input_ids, attention_mask, self._no_token_ids
        )
        
        # Normalize with numerical stability
        max_log_prob = max(yes_log_prob, no_log_prob)
        yes_exp = math.exp(yes_log_prob - max_log_prob)
        no_exp = math.exp(no_log_prob - max_log_prob)
        total = yes_exp + no_exp
        
        return yes_exp / total
    
    def evaluate_imageability(self, idiom: str) -> Tuple[float, Dict[str, float]]:
        """
        Evaluate imageability using 14-question checklist.
        
        Returns:
            Tuple of (average_score, per_question_scores_dict)
        """
        question_scores = {}
        
        for qid, question_template in IMAGEABILITY_QUESTIONS:
            question = question_template.format(idiom=idiom)
            prompt = self._format_chat_prompt(IMAGEABILITY_SYSTEM_PROMPT, question)
            score = self._compute_yes_prob(prompt)
            question_scores[qid] = round(score, 6)
        
        # Arithmetic mean
        avg_score = sum(question_scores.values()) / len(question_scores)
        
        return avg_score, question_scores
    
    def evaluate_transparency(self, idiom: str, meaning: str) -> Tuple[float, Dict[str, float]]:
        """
        Evaluate transparency using 15-question checklist.
        
        Returns:
            Tuple of (average_score, per_question_scores_dict)
        """
        question_scores = {}
        
        for qid, question_template in TRANSPARENCY_QUESTIONS:
            question = question_template.format(idiom=idiom, meaning=meaning)
            prompt = self._format_chat_prompt(TRANSPARENCY_SYSTEM_PROMPT, question)
            score = self._compute_yes_prob(prompt)
            question_scores[qid] = round(score, 6)
        
        # Arithmetic mean
        avg_score = sum(question_scores.values()) / len(question_scores)
        
        return avg_score, question_scores
    
    def run(self) -> None:
        """Run the full evaluation pipeline."""
        print("=" * 70)
        print("PHASE 0 CHECKLIST-BASED EVALUATION")
        print(f"  Imageability: {len(IMAGEABILITY_QUESTIONS)} questions")
        print(f"  Transparency: {len(TRANSPARENCY_QUESTIONS)} questions")
        print("=" * 70)
        
        # Load data
        print("\n[1/5] Loading data...")
        self._load_idioms()
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
            for i, idiom_data in enumerate(tqdm(pending_idioms, desc="Evaluating")):
                idiom_id = idiom_data["idiom_id"]
                idiom_text = idiom_data["idiom"]
                definition = idiom_data["definition"]
                
                # Evaluate imageability (14 questions)
                img_score, img_details = self.evaluate_imageability(idiom_text)
                
                # Evaluate transparency (15 questions)
                trans_score, trans_details = self.evaluate_transparency(idiom_text, definition)
                
                # Store result
                self._results[idiom_id] = {
                    "idiom_id": idiom_id,
                    "idiom": idiom_text,
                    "definition": definition,
                    "imageability": round(img_score, 6),
                    "transparency": round(trans_score, 6),
                    "imageability_details": img_details,
                    "transparency_details": trans_details,
                }
                
                # Debug output
                if self.test_mode or i < 3:
                    print(f"\n  [{idiom_id}] {idiom_text}")
                    print(f"      imageability={img_score:.4f} (avg of 14 questions)")
                    print(f"      transparency={trans_score:.4f} (avg of 15 questions)")
                    if self.test_mode:
                        print(f"      Imageability details: {img_details}")
                        print(f"      Transparency details: {trans_details}")
                
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
        description="Run Phase0 checklist-based evaluation (imageability + transparency).",
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
        help="Device to run model on (default: cuda).",
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
    
    timestamp = get_timestamp()
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
        evaluator = ChecklistEvaluator(
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
