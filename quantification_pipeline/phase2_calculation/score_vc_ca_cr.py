from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

# Add project root to path for centralized config import
_THIS_DIR = Path(__file__).parent.resolve()
_PROJECT_ROOT = _THIS_DIR.parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# Import centralized path configuration
from project_config import get_model_output_dir


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data: Dict[str, Any]) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def get_track(data: Dict[str, Any]) -> Dict[str, Any]:
    if "figurative_track" in data:
        return data.get("figurative_track", {}) or {}
    if "literal_track" in data:
        return data.get("literal_track", {}) or {}
    return {}


def build_id_score_map(track: Dict[str, Any]) -> Dict[str, float]:
    mapping: Dict[str, float] = {}
    for entity in track.get("entities", []) or []:
        entity_id = entity.get("id")
        if entity_id:
            mapping[entity_id] = float(entity.get("score", 0.0))
    for action in track.get("actions", []) or []:
        action_id = action.get("id")
        if action_id:
            mapping[action_id] = float(action.get("score", 0.0))
    return mapping


def compute_vc(track: Dict[str, Any]) -> float:
    relationships = track.get("relationships", []) or []
    if not relationships:
        return 0.0

    id_score = build_id_score_map(track)
    best = 0.0
    for rel in relationships:
        subject_id = rel.get("subject_id")
        action_id = rel.get("action_id")
        object_id = rel.get("object_id")

        subject_score = id_score.get(subject_id, 0.0)
        action_score = id_score.get(action_id, 0.0)

        if object_id is None:
            product = subject_score * action_score
        else:
            object_score = id_score.get(object_id, 0.0)
            product = subject_score * action_score * object_score

        if product > best:
            best = product

    return float(best)


def compute_ca(track: Dict[str, Any]) -> float:
    elements: List[Dict[str, Any]] = []
    elements.extend(track.get("entities", []) or [])
    elements.extend(track.get("actions", []) or [])

    if not elements:
        return 0.0

    weighted_sum = 0.0
    weight_total = 0.0
    for elem in elements:
        requires_cultural_context = bool(elem.get("requires_cultural_context", False))
        weight = 0.0 if requires_cultural_context else 1.0
        score = float(elem.get("score", 0.0))
        weighted_sum += weight * score
        weight_total += weight

    if weight_total == 0.0:
        return 0.0
    return float(weighted_sum / weight_total)


def compute_cr(literal_track: Dict[str, Any]) -> float:
    elements: List[Dict[str, Any]] = []
    elements.extend(literal_track.get("entities", []) or [])
    elements.extend(literal_track.get("actions", []) or [])

    if not elements:
        return 0.0

    total = 0.0
    for elem in elements:
        total += float(elem.get("score", 0.0))
    return float(total / len(elements))


def iter_image_dirs(base_dir: Path) -> Iterable[Path]:
    for idiom_dir in sorted(base_dir.glob("idiom_*")):
        if not idiom_dir.is_dir():
            continue
        for image_dir in sorted(idiom_dir.glob("image_*")):
            if image_dir.is_dir():
                yield image_dir


def process_image_dir(image_dir: Path) -> Tuple[bool, Optional[str]]:
    figurative_path = image_dir / "figurative.json"
    literal_path = image_dir / "literal.json"

    if not figurative_path.exists() or not literal_path.exists():
        return False, f"missing figurative.json or literal.json in {image_dir}"

    figurative_data = load_json(figurative_path)
    literal_data = load_json(literal_path)

    figurative_track = get_track(figurative_data)
    literal_track = get_track(literal_data)

    figurative_score = {
        "VC": compute_vc(figurative_track),
        "CA": compute_ca(figurative_track),
    }

    literal_score = {
        "VC": compute_vc(literal_track),
        "CA": compute_ca(literal_track),
        "CR": compute_cr(literal_track),
    }

    write_json(image_dir / "figurative_score.json", figurative_score)
    write_json(image_dir / "literal_score.json", literal_score)

    return True, None


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compute VC/CA/CR scores for IRFL figurative and literal JSON files."
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=get_model_output_dir("qwen3_vl_2b", "T2IMVI"),
        help="Root directory containing idiom_<id>/image_<id> folders.",
    )
    args = parser.parse_args()

    base_dir = args.base_dir
    if not base_dir.exists():
        print(f"Base dir not found: {base_dir}")
        return 1

    processed = 0
    skipped = 0
    for image_dir in iter_image_dirs(base_dir):
        ok, err = process_image_dir(image_dir)
        if ok:
            processed += 1
        else:
            skipped += 1
            print(f"Skip: {err}")

    print(f"Done. processed={processed}, skipped={skipped}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
