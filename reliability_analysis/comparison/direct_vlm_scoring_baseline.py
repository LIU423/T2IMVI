"""
Direct VLM scoring baseline for reliability comparison experiments.

This script scores idiom-image pairs directly with a VLM using a single
comparison prompt (no extracted entities/actions), then saves results in
`total_score.json` format so reliability_analysis can consume them directly.
"""

from __future__ import annotations

import argparse
import ast
import importlib.util
import json
import logging
import math
import re
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image

from project_config import (
    INPUT_IRFL_MATCHED_IMAGES_DIR,
    OUTPUT_PHASE0_DIR,
    PROMPT_DIR,
    RELIABILITY_ANALYSIS_COMPARISON_DIR,
)
from quantification_pipeline.qwen3_vl_loader import (
    get_qwen3_vl_generation_model_class,
    is_qwen3_vl_moe_model,
)


logger = logging.getLogger(__name__)

MODEL_NAME_TO_ID: Dict[str, str] = {
    "qwen3-vl-2b": "Qwen/Qwen3-VL-2B-Instruct",
    "qwen3-vl-30b-a3b-instruct": "Qwen/Qwen3-VL-30B-A3B-Instruct",
}

MODEL_NAME_TO_DIR: Dict[str, str] = {
    "qwen3-vl-2b": "qwen3_vl_2b",
    "qwen3-vl-30b-a3b-instruct": "qwen3_vl_30b_a3b_instruct",
}

DEFAULT_PROMPT_FILE = PROMPT_DIR / "comparison" / "direct_vlm_scoring_baseline.txt"
DEFAULT_OUTPUT_ROOT = RELIABILITY_ANALYSIS_COMPARISON_DIR / "direct_vlm_scoring_baseline"
DEFAULT_S_POT_K = 9.0


@dataclass
class ScoreRecord:
    idiom_id: int
    image_id: int
    idiom: str
    total_score: float
    evidence: List[str]
    raw_response: str


class Qwen3VLDirectScorer:
    """Generate direct idiom-image scores with Qwen3-VL."""

    def __init__(
        self,
        model_name: str,
        device: str,
        torch_dtype: str,
        max_new_tokens: int,
    ) -> None:
        self.model_name = model_name
        self.model_id = MODEL_NAME_TO_ID[model_name]
        self.device = device
        self.torch_dtype = torch_dtype
        self.max_new_tokens = max_new_tokens
        self.model = None
        self.processor = None

    def _resolve_torch_dtype(self):
        import torch

        mapping = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
            "auto": "auto",
        }
        return mapping[self.torch_dtype]

    def load(self) -> None:
        if self.model is not None:
            return

        import torch

        if is_qwen3_vl_moe_model(self.model_id):
            from transformers import AutoModelForImageTextToText, AutoProcessor

            self.processor = AutoProcessor.from_pretrained(
                self.model_id,
                trust_remote_code=True,
            )
            self.model = AutoModelForImageTextToText.from_pretrained(
                self.model_id,
                dtype=self._resolve_torch_dtype(),
                device_map=self.device,
                trust_remote_code=True,
            ).eval()
        else:
            from transformers import (
                AutoImageProcessor,
                AutoTokenizer,
                Qwen3VLProcessor,
                Qwen3VLVideoProcessor,
            )

            generation_model_class = get_qwen3_vl_generation_model_class(self.model_id)
            tokenizer = AutoTokenizer.from_pretrained(
                self.model_id,
                trust_remote_code=True,
            )
            image_processor = AutoImageProcessor.from_pretrained(
                self.model_id,
                trust_remote_code=True,
            )
            video_processor = Qwen3VLVideoProcessor.from_pretrained(
                self.model_id,
                trust_remote_code=True,
            )
            self.processor = Qwen3VLProcessor(
                image_processor=image_processor,
                tokenizer=tokenizer,
                video_processor=video_processor,
            )
            if hasattr(tokenizer, "chat_template") and tokenizer.chat_template:
                self.processor.chat_template = tokenizer.chat_template

            self.model = generation_model_class.from_pretrained(
                self.model_id,
                torch_dtype=self._resolve_torch_dtype(),
                device_map=self.device,
                trust_remote_code=True,
            ).eval()

        logger.info("Loaded model: %s", self.model_id)

    def unload(self) -> None:
        if self.model is not None:
            del self.model
            self.model = None
        if self.processor is not None:
            del self.processor
            self.processor = None
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    def _extract_json_object(self, text: str) -> Dict[str, Any]:
        fenced = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
        candidate = fenced.group(1).strip() if fenced else text.strip()

        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass

        start = candidate.find("{")
        end = candidate.rfind("}")
        if start >= 0 and end > start:
            return json.loads(candidate[start : end + 1])

        raise ValueError("Cannot parse JSON from model response")

    def score(
        self,
        image: Image.Image,
        idiom: str,
        prompt_text: str,
        max_parse_retries: int,
    ) -> ScoreRecord:
        if self.model is None or self.processor is None:
            raise RuntimeError("Model is not loaded")

        prompt = prompt_text.strip()
        repair_hint = ""
        last_error: Optional[Exception] = None

        for _ in range(max_parse_retries):
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": prompt + repair_hint},
                    ],
                }
            ]
            inputs = self.processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
            )
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            output = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
            )

            gen_ids = output[:, inputs["input_ids"].shape[-1] :]
            raw = self.processor.batch_decode(gen_ids, skip_special_tokens=True)[0].strip()

            try:
                parsed = self._extract_json_object(raw)
                score = float(parsed["total_score"])
                evidence = parsed.get("evidence", [])
                if not isinstance(evidence, list):
                    evidence = []
                evidence = [str(x) for x in evidence[:3]]
                score = max(0.0, min(1.0, score))
                idiom_echo = str(parsed.get("idiom", idiom))
                return ScoreRecord(
                    idiom_id=-1,
                    image_id=-1,
                    idiom=idiom_echo,
                    total_score=score,
                    evidence=evidence,
                    raw_response=raw,
                )
            except Exception as exc:
                last_error = exc
                repair_hint = (
                    "\n\nIMPORTANT: Return STRICT JSON ONLY with keys "
                    '{"idiom","total_score","evidence"} and valid numeric score in [0,1].'
                )

        raise ValueError(f"Failed to parse model response: {last_error}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Direct VLM scoring baseline for reliability comparison",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="qwen3-vl-2b",
        choices=sorted(MODEL_NAME_TO_ID.keys()),
        help="Model key",
    )
    parser.add_argument(
        "--idiom-ids",
        type=int,
        nargs="+",
        default=None,
        help="Specific idiom ids to process. Default: all available idioms.",
    )
    parser.add_argument(
        "--max-images-per-idiom",
        type=int,
        default=None,
        help="Optional image cap per idiom",
    )
    parser.add_argument(
        "--prompt-file",
        type=Path,
        default=DEFAULT_PROMPT_FILE,
        help="Path to direct scoring prompt file",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Root output dir for comparison runs",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="HuggingFace device_map",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="auto",
        choices=["auto", "float16", "bfloat16", "float32"],
        help="Torch dtype",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=220,
        help="Generation max_new_tokens",
    )
    parser.add_argument(
        "--max-parse-retries",
        type=int,
        default=2,
        help="Retries for malformed JSON responses",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing outputs",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Verbose logging",
    )
    return parser.parse_args()


def setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def _load_prompt_spec(prompt_file: Path) -> Dict[str, Any]:
    """
    Load prompt spec from .txt or .py.

    Text template:
    - Supports `<idiom>` or `{idiom}` placeholders.
    - If no placeholder exists, idiom is appended as `Idiom: ...`.

    Python template:
    - Must define either `build_prompt(idiom: str) -> str`
      or `PROMPT_TEMPLATE` (supports `{idiom}` formatting).
    """
    prompt_file = Path(prompt_file)
    if not prompt_file.exists():
        raise FileNotFoundError(f"Prompt file not found: {prompt_file}")

    if prompt_file.suffix.lower() == ".py":
        spec = importlib.util.spec_from_file_location("direct_vlm_prompt_module", prompt_file)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Cannot import prompt module: {prompt_file}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        build_fn = getattr(module, "build_prompt", None)
        if callable(build_fn):
            return {"type": "python_build_fn", "build_prompt": build_fn}

        template = getattr(module, "PROMPT_TEMPLATE", None)
        if isinstance(template, str):
            return {"type": "template", "template": template}

        raise ValueError(
            "Python prompt file must define build_prompt(idiom) or PROMPT_TEMPLATE"
        )

    with open(prompt_file, "r", encoding="utf-8") as f:
        return {"type": "template", "template": f.read().strip()}


def _render_prompt(prompt_spec: Dict[str, Any], idiom: str) -> str:
    """Render final prompt text for one idiom."""
    if prompt_spec["type"] == "python_build_fn":
        return str(prompt_spec["build_prompt"](idiom))

    template = str(prompt_spec["template"]).strip()
    if "<idiom>" in template:
        return template.replace("<idiom>", idiom)
    if "{idiom}" in template:
        return template.format(idiom=idiom)
    return f"{template}\n\nIdiom: {idiom}"


def _parse_phrase_from_metadata(path: Path) -> Optional[str]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None

    phrase = payload.get("phrase")
    if isinstance(phrase, str) and phrase.strip():
        return phrase.strip()
    return None


def _parse_definition_list(value: Any) -> List[str]:
    if isinstance(value, list):
        return [str(x) for x in value]
    if isinstance(value, str):
        try:
            parsed = ast.literal_eval(value)
            if isinstance(parsed, list):
                return [str(x) for x in parsed]
        except (ValueError, SyntaxError):
            return [value]
    return []


def _fallback_idiom_string(idiom_id: int, idiom_dir: Path) -> str:
    json_files = sorted(idiom_dir.glob("*.json"))
    for meta in json_files:
        try:
            with open(meta, "r", encoding="utf-8") as f:
                payload = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue
        phrase = payload.get("phrase")
        if isinstance(phrase, str) and phrase.strip():
            return phrase.strip()
        defs = _parse_definition_list(payload.get("definition"))
        if defs:
            return defs[0]
    return f"idiom_{idiom_id}"


def _load_phase0_score_map() -> Dict[int, Dict[str, float]]:
    score_map: Dict[int, Dict[str, float]] = {}
    phase0_files = sorted(
        OUTPUT_PHASE0_DIR.glob("phase0*.json"),
        key=lambda p: p.stat().st_mtime,
    )
    for phase0_path in phase0_files:
        try:
            with open(phase0_path, "r", encoding="utf-8") as f:
                payload = json.load(f)
        except (OSError, json.JSONDecodeError, ValueError):
            continue
        if not isinstance(payload, list):
            continue
        for item in payload:
            if not isinstance(item, dict):
                continue
            idiom_id = item.get("idiom_id")
            a1 = item.get("imageability")
            a2 = item.get("transparency")
            if isinstance(idiom_id, int) and isinstance(a1, (int, float)) and isinstance(a2, (int, float)):
                score_map[idiom_id] = {
                    "imageability": float(a1),
                    "transparency": float(a2),
                }
    return score_map


def _compute_s_pot(a1: float, a2: float, b1: float, b2: float, k: float = DEFAULT_S_POT_K) -> float:
    p = 1.0 + k * (1.0 - a2)
    mean_power = ((b1**p + b2**p) / 2.0) ** (1.0 / p)
    return (1.0 - a1) * b1 + a1 * mean_power


def _compute_s_fid(a1: float, a2: float, b1: float, b2: float) -> float:
    w_fig = 1.0 - 0.5 * a2
    w_lit = 0.5 * a2
    return (1.0 - a1) * b1 + a1 * (w_fig * b1 + w_lit * b2)


def _discover_idiom_ids() -> List[int]:
    idiom_ids: List[int] = []
    for path in INPUT_IRFL_MATCHED_IMAGES_DIR.iterdir():
        if not path.is_dir():
            continue
        try:
            idiom_ids.append(int(path.name))
        except ValueError:
            continue
    return sorted(idiom_ids)


def _discover_images_for_idiom(idiom_id: int) -> List[Tuple[int, Path]]:
    idiom_dir = INPUT_IRFL_MATCHED_IMAGES_DIR / str(idiom_id)
    images: List[Tuple[int, Path]] = []
    if not idiom_dir.exists():
        return images
    for image_path in idiom_dir.iterdir():
        if image_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        try:
            image_id = int(image_path.stem)
        except ValueError:
            continue
        images.append((image_id, image_path))
    return sorted(images, key=lambda x: x[0])


def _build_total_score_payload(
    idiom_id: int,
    image_id: int,
    direct_score: float,
    phase0_scores: Optional[Dict[str, float]],
) -> Dict[str, Any]:
    if phase0_scores is None:
        a1, a2 = 0.0, 0.0
    else:
        a1 = float(phase0_scores["imageability"])
        a2 = float(phase0_scores["transparency"])

    b1 = direct_score
    b2 = direct_score
    return {
        "idiom_id": idiom_id,
        "image_id": image_id,
        "imageability": a1,
        "transparency": a2,
        "figurative_score": b1,
        "literal_score": b2,
        "S_pot": _compute_s_pot(a1, a2, b1, b2),
        "S_fid": _compute_s_fid(a1, a2, b1, b2),
    }


def run(args: argparse.Namespace) -> None:
    prompt_spec = _load_prompt_spec(args.prompt_file)

    model_dir = MODEL_NAME_TO_DIR[args.model]
    output_base = args.output_root / model_dir
    output_base.mkdir(parents=True, exist_ok=True)

    idiom_ids = args.idiom_ids if args.idiom_ids else _discover_idiom_ids()
    phase0_score_map = _load_phase0_score_map()

    scorer = Qwen3VLDirectScorer(
        model_name=args.model,
        device=args.device,
        torch_dtype=args.dtype,
        max_new_tokens=args.max_new_tokens,
    )
    scorer.load()

    processed = 0
    skipped = 0
    failed = 0
    timing_total_seconds = 0.0
    timing_per_idiom: Dict[int, Dict[str, float]] = {}

    try:
        for idiom_id in idiom_ids:
            idiom_dir = INPUT_IRFL_MATCHED_IMAGES_DIR / str(idiom_id)
            if not idiom_dir.exists():
                logger.warning("Idiom dir missing: %s", idiom_dir)
                continue

            images = _discover_images_for_idiom(idiom_id)
            if args.max_images_per_idiom is not None:
                images = images[: args.max_images_per_idiom]
            if not images:
                logger.warning("No images found for idiom_%s", idiom_id)
                continue

            idiom_text = _fallback_idiom_string(idiom_id, idiom_dir)
            logger.info("Processing idiom_%s (%s images)", idiom_id, len(images))

            for image_id, image_path in images:
                out_dir = output_base / f"idiom_{idiom_id}" / f"image_{image_id}"
                total_path = out_dir / "total_score.json"
                detail_path = out_dir / "direct_vlm_baseline.json"

                if total_path.exists() and detail_path.exists() and not args.overwrite:
                    skipped += 1
                    continue

                metadata_path = image_path.with_suffix(".json")
                maybe_phrase = _parse_phrase_from_metadata(metadata_path)
                idiom_text_for_image = maybe_phrase or idiom_text
                prompt_text = _render_prompt(prompt_spec, idiom_text_for_image)

                try:
                    t0 = time.perf_counter()
                    image = Image.open(image_path).convert("RGB")
                    record = scorer.score(
                        image=image,
                        idiom=idiom_text_for_image,
                        prompt_text=prompt_text,
                        max_parse_retries=args.max_parse_retries,
                    )
                    record.idiom_id = idiom_id
                    record.image_id = image_id

                    out_dir.mkdir(parents=True, exist_ok=True)

                    payload = _build_total_score_payload(
                        idiom_id=idiom_id,
                        image_id=image_id,
                        direct_score=record.total_score,
                        phase0_scores=phase0_score_map.get(idiom_id),
                    )
                    with open(total_path, "w", encoding="utf-8") as f:
                        json.dump(payload, f, indent=2, ensure_ascii=False)

                    detail = {
                        "idiom_id": idiom_id,
                        "image_id": image_id,
                        "idiom": record.idiom,
                        "total_score": record.total_score,
                        "evidence": record.evidence,
                        "model_name": args.model,
                        "model_id": MODEL_NAME_TO_ID[args.model],
                    }
                    with open(detail_path, "w", encoding="utf-8") as f:
                        json.dump(detail, f, indent=2, ensure_ascii=False)

                    raw_path = out_dir / "raw_response.txt"
                    with open(raw_path, "w", encoding="utf-8") as f:
                        f.write(record.raw_response + "\n")

                    elapsed = time.perf_counter() - t0
                    timing_total_seconds += elapsed
                    idiom_timing = timing_per_idiom.setdefault(
                        idiom_id,
                        {"processed_images": 0, "total_seconds": 0.0},
                    )
                    idiom_timing["processed_images"] += 1
                    idiom_timing["total_seconds"] += elapsed
                    processed += 1
                except Exception as exc:
                    failed += 1
                    logger.error(
                        "Failed idiom_%s image_%s (%s): %s",
                        idiom_id,
                        image_id,
                        image_path.name,
                        exc,
                    )
    finally:
        scorer.unload()

    logger.info(
        "Done. processed=%s skipped=%s failed=%s output=%s",
        processed,
        skipped,
        failed,
        output_base,
    )
    avg_per_image = (timing_total_seconds / processed) if processed else 0.0
    idiom_timing_summary: Dict[str, Dict[str, float]] = {}
    for idiom_id, item in timing_per_idiom.items():
        n = int(item["processed_images"])
        total_s = float(item["total_seconds"])
        idiom_timing_summary[str(idiom_id)] = {
            "processed_images": n,
            "total_seconds": total_s,
            "avg_seconds_per_image": (total_s / n) if n else 0.0,
        }
    timing_summary = {
        "phase": "comparison_direct_vlm_baseline",
        "model_name": args.model,
        "model_id": MODEL_NAME_TO_ID[args.model],
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "output_dir": str(output_base),
        "processed_images": processed,
        "skipped_images": skipped,
        "failed_images": failed,
        "total_seconds": timing_total_seconds,
        "avg_seconds_per_image": avg_per_image,
        "per_idiom": idiom_timing_summary,
    }
    timing_path = output_base / "timing_direct_vlm_baseline.json"
    with open(timing_path, "w", encoding="utf-8") as f:
        json.dump(timing_summary, f, indent=2, ensure_ascii=False)


def main() -> None:
    args = parse_args()
    setup_logging(args.verbose)
    run(args)


if __name__ == "__main__":
    main()
