from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

# Add project root to path for centralized config import
_THIS_DIR = Path(__file__).parent.resolve()
_PROJECT_ROOT = _THIS_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# Import centralized path configuration
from project_config import (
    OUTPUT_IRFL_DIR,
    OUTPUT_PHASE0_DIR,
    get_model_output_dir,
    get_phase0_output_file,
)


def parse_idiom_ids(raw_values: Optional[List[str]]) -> Optional[set[int]]:
    if not raw_values:
        return None

    idiom_ids: set[int] = set()
    for raw in raw_values:
        for token in raw.split(","):
            token = token.strip()
            if not token:
                continue
            try:
                idiom_ids.add(int(token))
            except ValueError as exc:
                raise argparse.ArgumentTypeError(f"Invalid idiom id: {token}") from exc

    return idiom_ids or None


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def mean_numeric_values(data: Dict[str, Any]) -> float:
    values: List[float] = []
    for value in data.values():
        if isinstance(value, (int, float)):
            values.append(float(value))
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def load_phase0_map(path: Path) -> Dict[int, Dict[str, float]]:
    items = load_json(path)
    mapping: Dict[int, Dict[str, float]] = {}
    for item in items:
        idiom_id = int(item.get("idiom_id"))
        mapping[idiom_id] = {
            "imageability": float(item.get("imageability", 0.0)),
            "transparency": float(item.get("transparency", 0.0)),
        }
    return mapping


def compute_s_pot(a1: float, a2: float, b1: float, b2: float, k: float) -> float:
    p = 1.0 + k * (1.0 - a2)
    mean_power = ((b1 ** p + b2 ** p) / 2.0) ** (1.0 / p)
    return (1.0 - a1) * b1 + a1 * mean_power


def compute_s_fid(a1: float, a2: float, b1: float, b2: float) -> float:
    w_fig = 1.0 - 0.5 * a2
    w_lit = 0.5 * a2
    return (1.0 - a1) * b1 + a1 * (w_fig * b1 + w_lit * b2)


def iter_idiom_dirs(base_dir: Path, idiom_ids: Optional[set[int]] = None) -> Iterable[Path]:
    for idiom_dir in sorted(base_dir.glob("idiom_*")):
        if not idiom_dir.is_dir():
            continue
        if idiom_ids is not None:
            idiom_id = parse_suffix_id(idiom_dir.name, "idiom_")
            if idiom_id is None or idiom_id not in idiom_ids:
                continue
        yield idiom_dir


def iter_image_dirs(idiom_dir: Path) -> Iterable[Path]:
    for image_dir in sorted(idiom_dir.glob("image_*")):
        if image_dir.is_dir():
            yield image_dir


def parse_suffix_id(name: str, prefix: str) -> Optional[int]:
    if not name.startswith(prefix):
        return None
    tail = name[len(prefix):]
    if not tail.isdigit():
        return None
    return int(tail)


def process_image_dir(
    image_dir: Path,
    idiom_id: int,
    image_id: int,
    phase0_map: Dict[int, Dict[str, float]],
    k: float,
    image_output_name: str,
) -> Tuple[bool, Optional[Dict[str, float]], Optional[str]]:
    figurative_path = image_dir / "figurative_score.json"
    literal_path = image_dir / "literal_score.json"

    if not figurative_path.exists() or not literal_path.exists():
        return False, None, f"missing figurative_score.json or literal_score.json in {image_dir}"

    if idiom_id not in phase0_map:
        return False, None, f"missing phase0 scores for idiom_id={idiom_id}"

    figurative_data = load_json(figurative_path)
    literal_data = load_json(literal_path)

    b1 = mean_numeric_values(figurative_data)
    b2 = mean_numeric_values(literal_data)

    a1 = phase0_map[idiom_id]["imageability"]
    a2 = phase0_map[idiom_id]["transparency"]

    s_pot = compute_s_pot(a1, a2, b1, b2, k)
    s_fid = compute_s_fid(a1, a2, b1, b2)

    payload = {
        "idiom_id": idiom_id,
        "image_id": image_id,
        "imageability": a1,
        "transparency": a2,
        "figurative_score": b1,
        "literal_score": b2,
        "S_pot": s_pot,
        "S_fid": s_fid,
    }

    write_json(image_dir / image_output_name, payload)
    return True, payload, None


def average_metrics(items: Iterable[Dict[str, float]], keys: Iterable[str]) -> Dict[str, float]:
    values: Dict[str, List[float]] = {key: [] for key in keys}
    for item in items:
        for key in keys:
            if key in item:
                values[key].append(float(item[key]))

    averages: Dict[str, float] = {}
    for key, nums in values.items():
        averages[key] = float(sum(nums) / len(nums)) if nums else 0.0
    return averages


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compute total scores for IRFL images and aggregate by idiom."
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=get_model_output_dir("qwen3_vl_2b", "T2IMVI"),
        help="Root directory containing idiom_<id>/image_<id> folders.",
    )
    parser.add_argument(
        "--phase0-json",
        type=Path,
        default=OUTPUT_PHASE0_DIR / "phase0_test.json",
        help="Phase0 JSON containing imageability/transparency per idiom_id.",
    )
    parser.add_argument(
        "--k",
        type=float,
        default=9.0,
        help="k parameter in p = 1 + k * (1 - a2).",
    )
    parser.add_argument(
        "--image-output-name",
        type=str,
        default="total_score.json",
        help="Filename to write per-image total score JSON.",
    )
    parser.add_argument(
        "--idiom-output-name",
        type=str,
        default="idiom_total_scores.json",
        help="Filename to write per-idiom aggregation JSON.",
    )
    parser.add_argument(
        "--base-output-name",
        type=str,
        default="all_idiom_total_scores.json",
        help="Filename to write base-dir aggregation JSON.",
    )
    parser.add_argument(
        "--idiom-ids",
        nargs="+",
        default=None,
        help="Optional idiom IDs to process. Supports space-separated and comma-separated values.",
    )
    args = parser.parse_args()

    base_dir = args.base_dir
    idiom_ids = parse_idiom_ids(args.idiom_ids)
    if not base_dir.exists():
        print(f"Base dir not found: {base_dir}")
        return 1

    if not args.phase0_json.exists():
        print(f"Phase0 json not found: {args.phase0_json}")
        return 1

    phase0_map = load_phase0_map(args.phase0_json)

    processed_images = 0
    skipped_images = 0
    idiom_summaries: Dict[str, Dict[str, Any]] = {}

    for idiom_dir in iter_idiom_dirs(base_dir, idiom_ids=idiom_ids):
        idiom_id = parse_suffix_id(idiom_dir.name, "idiom_")
        if idiom_id is None:
            continue

        image_payloads: Dict[str, Dict[str, float]] = {}
        for image_dir in iter_image_dirs(idiom_dir):
            image_id = parse_suffix_id(image_dir.name, "image_")
            if image_id is None:
                continue

            ok, payload, err = process_image_dir(
                image_dir,
                idiom_id,
                image_id,
                phase0_map,
                args.k,
                args.image_output_name,
            )
            if ok and payload is not None:
                processed_images += 1
                image_payloads[image_dir.name] = payload
            else:
                skipped_images += 1
                print(f"Skip: {err}")

        averages = average_metrics(
            image_payloads.values(),
            ["figurative_score", "literal_score", "S_pot", "S_fid"],
        )

        idiom_summary = {
            "idiom_id": idiom_id,
            "images": image_payloads,
            "averages": averages,
        }
        write_json(idiom_dir / args.idiom_output_name, idiom_summary)
        idiom_summaries[idiom_dir.name] = idiom_summary

    base_averages = average_metrics(
        (summary["averages"] for summary in idiom_summaries.values()),
        ["figurative_score", "literal_score", "S_pot", "S_fid"],
    )

    base_summary = {
        "idioms": idiom_summaries,
        "averages": base_averages,
    }
    write_json(base_dir / args.base_output_name, base_summary)

    print(
        f"Done. processed_images={processed_images}, skipped_images={skipped_images}, "
        f"idioms={len(idiom_summaries)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
