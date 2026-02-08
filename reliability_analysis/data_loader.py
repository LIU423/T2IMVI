"""
Data loading module for T2IMVI Reliability Analysis.

This module provides functions to load human annotations and model outputs
from the file system. It handles the specific JSON formats used in the
IRFL dataset and T2IMVI model outputs.
"""

import json
import ast
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any, Iterator
import logging

from config import (
    INPUT_PATH,
    OUTPUT_PATH,
    get_model_config,
    get_annotation_path,
    get_model_output_path,
    ModelConfig,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class Annotation:
    """Represents a human annotation for an image.
    
    Attributes:
        irfl_id: Unique IRFL identifier
        idiom_id: The idiom ID (parsed from directory structure)
        image_id: The image ID
        annotations: List of 5 annotation labels from annotators
        category: Aggregated category (may be NaN)
        phrase: The idiom phrase
        query: The search query used
        literal_candidate: Whether image is a literal candidate
        definition: List of definitions for the phrase
        figurative_type: Type of figurative language (e.g., 'idiom')
    """
    irfl_id: str
    idiom_id: int
    image_id: int
    annotations: List[str]
    category: Optional[str]
    phrase: str
    query: str
    literal_candidate: bool
    definition: List[str]
    figurative_type: str
    
    @property
    def annotation_counts(self) -> Dict[str, int]:
        """Get counts of each annotation label."""
        counts: Dict[str, int] = {}
        for label in self.annotations:
            counts[label] = counts.get(label, 0) + 1
        return counts


@dataclass
class ModelOutput:
    """Represents model output scores for an image.
    
    Attributes:
        idiom_id: The idiom ID
        image_id: The image ID
        imageability: Imageability score
        transparency: Transparency score
        figurative_score: Score for figurative interpretation
        literal_score: Score for literal interpretation
        s_pot: Potential score
        s_fid: Fidelity score
        entity_action_avg: Mean score across figurative/literal entities and actions
    """
    idiom_id: int
    image_id: int
    imageability: float
    transparency: float
    figurative_score: float
    literal_score: float
    s_pot: float
    s_fid: float
    entity_action_avg: float
    
    def get_score(self, score_field: str = "figurative_score") -> float:
        """Get a specific score by field name.
        
        Args:
            score_field: Name of the score field to retrieve.
                         Special values:
                         - "fig_lit_avg": arithmetic mean of figurative_score and literal_score
                         - "entity_action_avg": mean of entity/action scores from figurative/literal tracks
            
        Returns:
            The score value
            
        Raises:
            AttributeError: If score_field doesn't exist
        """
        if score_field == "fig_lit_avg":
            return (self.figurative_score + self.literal_score) / 2.0
        return getattr(self, score_field)


def _collect_entity_action_scores(track_data: Dict[str, Any]) -> List[float]:
    """Collect numeric scores from entities and actions in a track."""
    scores: List[float] = []
    for item in track_data.get("entities", []):
        score = item.get("score")
        if isinstance(score, (int, float)):
            scores.append(float(score))
    for item in track_data.get("actions", []):
        score = item.get("score")
        if isinstance(score, (int, float)):
            scores.append(float(score))
    return scores


def compute_entity_action_avg(
    model_key: str,
    idiom_id: int,
    image_id: int,
) -> float:
    """Compute mean score across figurative/literal entities and actions.

    Returns 0.0 if no valid scores are found.
    """
    config = get_model_config(model_key)
    image_dir = config.get_output_path() / f"idiom_{idiom_id}" / f"image_{image_id}"
    scores: List[float] = []

    track_sources = [
        (image_dir / "figurative.json", "figurative_track"),
        (image_dir / "literal.json", "literal_track"),
    ]

    for path, track_key in track_sources:
        if not path.exists():
            continue
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            track_data = data.get(track_key, {})
            scores.extend(_collect_entity_action_scores(track_data))
        except (json.JSONDecodeError, OSError, ValueError) as e:
            logger.warning(f"Failed to load track data {path}: {e}")

    if not scores:
        return 0.0
    return sum(scores) / len(scores)


@dataclass
class ImageData:
    """Combined annotation and model output for an image.
    
    Attributes:
        annotation: Human annotation data
        model_outputs: Dictionary mapping model_key to ModelOutput
    """
    annotation: Annotation
    model_outputs: Dict[str, ModelOutput]


# =============================================================================
# ANNOTATION LOADING
# =============================================================================

def parse_annotations_string(annotations_str: str) -> List[str]:
    """Parse the annotations string into a list of labels.
    
    The annotations field in IRFL is stored as a string representation
    of a Python list, e.g., "['Figurative', 'None', 'Literal']"
    
    Args:
        annotations_str: String representation of annotations list
        
    Returns:
        List of annotation labels
    """
    try:
        # Use ast.literal_eval for safe parsing of Python literal
        return ast.literal_eval(annotations_str)
    except (ValueError, SyntaxError) as e:
        logger.warning(f"Failed to parse annotations string: {annotations_str}")
        # Fallback: try basic string parsing
        cleaned = annotations_str.strip("[]").replace("'", "").replace('"', '')
        return [s.strip() for s in cleaned.split(",")]


def parse_definition_string(definition_str: str) -> List[str]:
    """Parse the definition string into a list of definitions.
    
    Args:
        definition_str: String representation of definitions list
        
    Returns:
        List of definition strings
    """
    try:
        return ast.literal_eval(definition_str)
    except (ValueError, SyntaxError):
        return [definition_str]


def load_annotation(idiom_id: int, image_id: int) -> Optional[Annotation]:
    """Load a single annotation from file.
    
    Args:
        idiom_id: The idiom ID
        image_id: The image ID
        
    Returns:
        Annotation object if file exists and is valid, None otherwise
    """
    path = get_annotation_path(idiom_id, image_id)
    
    if not path.exists():
        logger.debug(f"Annotation file not found: {path}")
        return None
    
    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Parse category - handle NaN
        category = data.get("category")
        if category is None or (isinstance(category, float) and str(category) == "nan"):
            category = None
        
        return Annotation(
            irfl_id=data["IRFL_id"],
            idiom_id=idiom_id,
            image_id=image_id,
            annotations=parse_annotations_string(data["annotations"]),
            category=category,
            phrase=data["phrase"],
            query=data["query"],
            literal_candidate=data.get("literal_candidate", False),
            definition=parse_definition_string(data.get("definition", "[]")),
            figurative_type=data.get("figurative_type", "unknown"),
        )
    except (json.JSONDecodeError, KeyError) as e:
        logger.warning(f"Failed to load annotation {path}: {e}")
        return None


def load_model_output(
    model_key: str, 
    idiom_id: int, 
    image_id: int
) -> Optional[ModelOutput]:
    """Load model output for a specific image.
    
    Args:
        model_key: Key in MODEL_CONFIGS dictionary
        idiom_id: The idiom ID
        image_id: The image ID
        
    Returns:
        ModelOutput object if file exists and is valid, None otherwise
    """
    path = get_model_output_path(model_key, idiom_id, image_id)
    
    if not path.exists():
        logger.debug(f"Model output file not found: {path}")
        return None
    
    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        entity_action_avg = compute_entity_action_avg(model_key, idiom_id, image_id)
        
        return ModelOutput(
            idiom_id=data["idiom_id"],
            image_id=data["image_id"],
            imageability=float(data.get("imageability", 0.0)),
            transparency=float(data.get("transparency", 0.0)),
            figurative_score=float(data.get("figurative_score", 0.0)),
            literal_score=float(data.get("literal_score", 0.0)),
            s_pot=float(data.get("S_pot", 0.0)),
            s_fid=float(data.get("S_fid", 0.0)),
            entity_action_avg=entity_action_avg,
        )
    except (json.JSONDecodeError, KeyError, ValueError) as e:
        logger.warning(f"Failed to load model output {path}: {e}")
        return None


# =============================================================================
# BATCH LOADING
# =============================================================================

def discover_idioms() -> List[int]:
    """Discover all available idiom IDs from the input directory.
    
    Returns:
        Sorted list of idiom IDs
    """
    idiom_ids = []
    for path in INPUT_PATH.iterdir():
        if path.is_dir():
            try:
                idiom_ids.append(int(path.name))
            except ValueError:
                continue
    return sorted(idiom_ids)


def discover_images_for_idiom(idiom_id: int) -> List[int]:
    """Discover all available image IDs for a given idiom.
    
    Args:
        idiom_id: The idiom ID
        
    Returns:
        Sorted list of image IDs
    """
    idiom_path = INPUT_PATH / str(idiom_id)
    if not idiom_path.exists():
        return []
    
    image_ids = []
    for path in idiom_path.glob("*.json"):
        try:
            image_ids.append(int(path.stem))
        except ValueError:
            continue
    return sorted(image_ids)


def load_all_annotations_for_idiom(idiom_id: int) -> List[Annotation]:
    """Load all annotations for a given idiom.
    
    Args:
        idiom_id: The idiom ID
        
    Returns:
        List of Annotation objects
    """
    annotations = []
    for image_id in discover_images_for_idiom(idiom_id):
        annotation = load_annotation(idiom_id, image_id)
        if annotation:
            annotations.append(annotation)
    return annotations


def load_all_model_outputs_for_idiom(
    model_key: str, 
    idiom_id: int
) -> Dict[int, ModelOutput]:
    """Load all model outputs for a given idiom.
    
    Args:
        model_key: Key in MODEL_CONFIGS dictionary
        idiom_id: The idiom ID
        
    Returns:
        Dictionary mapping image_id to ModelOutput
    """
    outputs = {}
    config = get_model_config(model_key)
    idiom_path = config.get_output_path() / f"idiom_{idiom_id}"
    
    if not idiom_path.exists():
        logger.warning(f"Idiom path not found for model {model_key}: {idiom_path}")
        return outputs
    
    for image_dir in idiom_path.iterdir():
        if image_dir.is_dir() and image_dir.name.startswith("image_"):
            try:
                image_id = int(image_dir.name.replace("image_", ""))
                output = load_model_output(model_key, idiom_id, image_id)
                if output:
                    outputs[image_id] = output
            except ValueError:
                continue
    
    return outputs


def load_combined_data_for_idiom(
    idiom_id: int,
    model_keys: Optional[List[str]] = None
) -> List[ImageData]:
    """Load combined annotation and model output data for an idiom.
    
    Args:
        idiom_id: The idiom ID
        model_keys: List of model keys to load (loads all if None)
        
    Returns:
        List of ImageData objects with matched annotations and outputs
    """
    from config import list_available_models
    
    if model_keys is None:
        model_keys = list_available_models()
    
    # Load all annotations
    annotations = load_all_annotations_for_idiom(idiom_id)
    
    # Load model outputs for each model
    model_outputs_by_model = {
        model_key: load_all_model_outputs_for_idiom(model_key, idiom_id)
        for model_key in model_keys
    }
    
    # Combine data
    combined = []
    for annotation in annotations:
        image_id = annotation.image_id
        model_outputs = {}
        
        for model_key in model_keys:
            if image_id in model_outputs_by_model[model_key]:
                model_outputs[model_key] = model_outputs_by_model[model_key][image_id]
        
        combined.append(ImageData(
            annotation=annotation,
            model_outputs=model_outputs,
        ))
    
    return combined


def iterate_all_data(
    model_keys: Optional[List[str]] = None,
    idiom_ids: Optional[List[int]] = None,
) -> Iterator[Tuple[int, List[ImageData]]]:
    """Iterate over all idioms and their combined data.
    
    Args:
        model_keys: List of model keys to load (loads all if None)
        idiom_ids: List of idiom IDs to process (all if None)
        
    Yields:
        Tuple of (idiom_id, List[ImageData])
    """
    if idiom_ids is None:
        idiom_ids = discover_idioms()
    
    for idiom_id in idiom_ids:
        data = load_combined_data_for_idiom(idiom_id, model_keys)
        if data:
            yield idiom_id, data


# =============================================================================
# VALIDATION UTILITIES
# =============================================================================

def validate_annotation(annotation: Annotation) -> List[str]:
    """Validate an annotation and return any issues found.
    
    Args:
        annotation: The annotation to validate
        
    Returns:
        List of validation error messages (empty if valid)
    """
    from config import DEFAULT_CLASSIFICATION_CONFIG
    
    errors = []
    
    # Check annotation count
    if len(annotation.annotations) != 5:
        errors.append(
            f"Expected 5 annotations, got {len(annotation.annotations)}"
        )
    
    # Check annotation values
    valid_labels = set(DEFAULT_CLASSIFICATION_CONFIG.all_labels)
    for label in annotation.annotations:
        if label not in valid_labels:
            errors.append(f"Unknown annotation label: {label}")
    
    return errors


def get_data_statistics(model_key: str) -> Dict[str, Any]:
    """Get statistics about available data for a model.
    
    Args:
        model_key: Key in MODEL_CONFIGS dictionary
        
    Returns:
        Dictionary with statistics
    """
    idiom_ids = discover_idioms()
    total_annotations = 0
    total_outputs = 0
    matched = 0
    
    for idiom_id in idiom_ids:
        annotations = load_all_annotations_for_idiom(idiom_id)
        outputs = load_all_model_outputs_for_idiom(model_key, idiom_id)
        
        total_annotations += len(annotations)
        total_outputs += len(outputs)
        
        annotation_ids = {a.image_id for a in annotations}
        output_ids = set(outputs.keys())
        matched += len(annotation_ids & output_ids)
    
    return {
        "model_key": model_key,
        "total_idioms": len(idiom_ids),
        "total_annotations": total_annotations,
        "total_model_outputs": total_outputs,
        "matched_pairs": matched,
        "coverage": matched / total_annotations if total_annotations > 0 else 0.0,
    }


if __name__ == "__main__":
    # Test data loading
    print("=" * 60)
    print("T2IMVI Data Loader - Test")
    print("=" * 60)
    
    # Discover idioms
    idiom_ids = discover_idioms()
    print(f"\nDiscovered {len(idiom_ids)} idioms")
    
    if idiom_ids:
        # Load first idiom's data
        first_idiom = idiom_ids[0]
        print(f"\nLoading data for idiom {first_idiom}...")
        
        annotations = load_all_annotations_for_idiom(first_idiom)
        print(f"  Loaded {len(annotations)} annotations")
        
        if annotations:
            first_ann = annotations[0]
            print(f"  First annotation: {first_ann.phrase}")
            print(f"  Annotations: {first_ann.annotations}")
            print(f"  Counts: {first_ann.annotation_counts}")
        
        # Try to load model outputs
        from config import list_available_models
        for model_key in list_available_models():
            outputs = load_all_model_outputs_for_idiom(model_key, first_idiom)
            print(f"  Model {model_key}: {len(outputs)} outputs")
