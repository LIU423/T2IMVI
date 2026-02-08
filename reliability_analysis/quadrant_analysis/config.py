"""Configuration for quadrant-based reliability analysis."""

from dataclasses import dataclass, field
from typing import List, Dict

# Score fields requested for comparison.
SCORE_FIELD_LABELS: Dict[str, str] = {
    "s_pot": "S_pot",
    "s_fid": "S_fid",
    "entity_action_avg": "Entity+Action Avg",
    "fig_lit_avg": "(Fig+Lit)/2",
}

DEFAULT_SCORE_FIELDS: List[str] = list(SCORE_FIELD_LABELS.keys())


@dataclass
class QuadrantConfig:
    """Configurable parameters for quadrant analysis."""

    transparency_threshold: float = 0.5
    imageability_threshold: float = 0.5
    rbo_p: float = 0.9
    score_fields: List[str] = field(default_factory=lambda: DEFAULT_SCORE_FIELDS.copy())
