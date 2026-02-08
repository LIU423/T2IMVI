"""
Failed-item logging utilities for phase1_scoring.
"""

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Set, Tuple


@dataclass
class FailedItemRecord:
    """Serializable failed item record."""

    timestamp: str
    model_name: str
    idiom_id: int
    image_num: int
    image_id: str
    is_oom: bool
    attempts: int
    error_type: str
    error_message: str


class FailedItemLogger:
    """Append-only JSONL logger for failed images."""

    def __init__(self, output_file: Path):
        self.output_file = Path(output_file)
        self.output_file.parent.mkdir(parents=True, exist_ok=True)

    def append(self, record: FailedItemRecord) -> None:
        payload = asdict(record)
        with open(self.output_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")

    @staticmethod
    def make_record(
        model_name: str,
        idiom_id: int,
        image_num: int,
        image_id: str,
        is_oom: bool,
        attempts: int,
        error: Exception,
    ) -> FailedItemRecord:
        return FailedItemRecord(
            timestamp=datetime.now(timezone.utc).isoformat(),
            model_name=model_name,
            idiom_id=idiom_id,
            image_num=image_num,
            image_id=image_id,
            is_oom=is_oom,
            attempts=attempts,
            error_type=type(error).__name__,
            error_message=str(error),
        )


def load_failed_items(paths: Iterable[Path]) -> List[Dict]:
    """Load JSONL failed-item records from one or more files."""
    records: List[Dict] = []
    for path in paths:
        p = Path(path)
        if not p.exists():
            continue
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                records.append(json.loads(line))
    return records


def extract_target_image_map(records: Iterable[Dict]) -> Dict[int, List[int]]:
    """Build {idiom_id: [image_num, ...]} map for rerun."""
    pairs: Set[Tuple[int, int]] = set()
    for record in records:
        idiom_id = int(record["idiom_id"])
        image_num = int(record["image_num"])
        pairs.add((idiom_id, image_num))

    by_idiom: Dict[int, List[int]] = {}
    for idiom_id, image_num in sorted(pairs):
        by_idiom.setdefault(idiom_id, []).append(image_num)
    return by_idiom
