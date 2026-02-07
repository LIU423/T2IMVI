"""
Classification module for T2IMVI Reliability Analysis.

This module implements the classification logic for categorizing images into
three classes based on human annotations:
- I_fig: Figurative images (majority Figurative or Figurative+Literal)
- I_lit: Literal images (majority Literal)
- I_rand: Random/mixed images (everything else)

Classification follows the priority ordering specified by the user:
Figurative+Literal ≈ Figurative > Literal ≈ Partial Literal > None
"""

from enum import Enum
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import logging

from config import (
    DEFAULT_CLASSIFICATION_CONFIG, 
    ClassificationConfig,
    DEFAULT_SCORING_CONFIG,
    ScoringConfig,
)
from data_loader import Annotation, ImageData

# Configure logging
logger = logging.getLogger(__name__)


# =============================================================================
# CLASSIFICATION ENUM
# =============================================================================

class ImageClass(Enum):
    """Classification categories for images.
    
    Human ground truth ranking: I_FIG > I_LIT > I_RAND
    """
    I_FIG = "I_fig"   # Figurative images
    I_LIT = "I_lit"   # Literal images
    I_RAND = "I_rand" # Random/mixed images
    
    def __lt__(self, other: "ImageClass") -> bool:
        """Define ordering for ranking: I_FIG > I_LIT > I_RAND."""
        order = {ImageClass.I_FIG: 0, ImageClass.I_LIT: 1, ImageClass.I_RAND: 2}
        return order[self] < order[other]
    
    def __le__(self, other: "ImageClass") -> bool:
        return self == other or self < other
    
    def __gt__(self, other: "ImageClass") -> bool:
        return not self <= other
    
    def __ge__(self, other: "ImageClass") -> bool:
        return not self < other


# =============================================================================
# CLASSIFICATION RESULT
# =============================================================================

@dataclass
class ClassificationResult:
    """Result of classifying an image.
    
    Attributes:
        image_class: The assigned class (I_FIG, I_LIT, I_RAND)
        confidence: Confidence score (fraction of annotations supporting this class)
        annotation_counts: Count of each annotation label
        reasoning: Human-readable explanation of the classification
    """
    image_class: ImageClass
    confidence: float
    annotation_counts: Dict[str, int]
    reasoning: str


# =============================================================================
# CLASSIFICATION LOGIC
# =============================================================================

def count_annotations_by_category(
    annotations: List[str],
    config: ClassificationConfig = DEFAULT_CLASSIFICATION_CONFIG
) -> Tuple[int, int, int]:
    """Count annotations by category (figurative, literal, other).
    
    Args:
        annotations: List of 5 annotation labels
        config: Classification configuration
        
    Returns:
        Tuple of (fig_count, lit_count, other_count)
    """
    fig_count = 0
    lit_count = 0
    other_count = 0
    
    for label in annotations:
        if label in config.fig_labels:
            fig_count += 1
        elif label in config.lit_labels:
            lit_count += 1
        else:
            other_count += 1
    
    return fig_count, lit_count, other_count


def classify_annotation(
    annotation: Annotation,
    config: ClassificationConfig = DEFAULT_CLASSIFICATION_CONFIG
) -> ClassificationResult:
    """Classify an annotation into I_fig, I_lit, or I_rand.
    
    Classification Rules (per user specification):
    1. If Figurative or Figurative+Literal is majority (≥3) → I_fig
    2. If Literal is majority (≥3), and Figurative+Literal is NOT majority → I_lit
    3. Otherwise → I_rand
    
    Args:
        annotation: The annotation to classify
        config: Classification configuration
        
    Returns:
        ClassificationResult with class, confidence, and reasoning
    """
    annotations = annotation.annotations
    counts = annotation.annotation_counts
    total = len(annotations)
    threshold = config.majority_threshold
    
    # Count by category
    fig_count, lit_count, other_count = count_annotations_by_category(
        annotations, config
    )
    
    # Classification logic
    if fig_count >= threshold:
        # Majority Figurative or Figurative+Literal → I_fig
        confidence = fig_count / total
        reasoning = (
            f"Figurative majority: {fig_count}/{total} annotations are "
            f"Figurative or Figurative+Literal"
        )
        return ClassificationResult(
            image_class=ImageClass.I_FIG,
            confidence=confidence,
            annotation_counts=counts,
            reasoning=reasoning,
        )
    
    elif lit_count >= threshold:
        # Majority Literal → I_lit
        confidence = lit_count / total
        reasoning = (
            f"Literal majority: {lit_count}/{total} annotations are Literal"
        )
        return ClassificationResult(
            image_class=ImageClass.I_LIT,
            confidence=confidence,
            annotation_counts=counts,
            reasoning=reasoning,
        )
    
    else:
        # No clear majority → I_rand
        max_count = max(fig_count, lit_count, other_count)
        confidence = max_count / total
        reasoning = (
            f"No majority: fig={fig_count}, lit={lit_count}, other={other_count}"
        )
        return ClassificationResult(
            image_class=ImageClass.I_RAND,
            confidence=confidence,
            annotation_counts=counts,
            reasoning=reasoning,
        )


def classify_image_data(
    image_data: ImageData,
    config: ClassificationConfig = DEFAULT_CLASSIFICATION_CONFIG
) -> ClassificationResult:
    """Classify an ImageData object.
    
    Args:
        image_data: Combined annotation and model output data
        config: Classification configuration
        
    Returns:
        ClassificationResult with class, confidence, and reasoning
    """
    return classify_annotation(image_data.annotation, config)


# =============================================================================
# BATCH CLASSIFICATION
# =============================================================================

@dataclass
class IdiomClassificationSummary:
    """Summary of classifications for an idiom.
    
    Attributes:
        idiom_id: The idiom ID
        total_images: Total number of images
        fig_images: List of image IDs classified as I_fig
        lit_images: List of image IDs classified as I_lit
        rand_images: List of image IDs classified as I_rand
        classifications: Mapping from image_id to ClassificationResult
    """
    idiom_id: int
    total_images: int
    fig_images: List[int]
    lit_images: List[int]
    rand_images: List[int]
    classifications: Dict[int, ClassificationResult]
    
    @property
    def fig_count(self) -> int:
        return len(self.fig_images)
    
    @property
    def lit_count(self) -> int:
        return len(self.lit_images)
    
    @property
    def rand_count(self) -> int:
        return len(self.rand_images)
    
    def get_images_by_class(self, image_class: ImageClass) -> List[int]:
        """Get image IDs for a specific class."""
        if image_class == ImageClass.I_FIG:
            return self.fig_images
        elif image_class == ImageClass.I_LIT:
            return self.lit_images
        else:
            return self.rand_images


def classify_idiom_images(
    image_data_list: List[ImageData],
    config: ClassificationConfig = DEFAULT_CLASSIFICATION_CONFIG
) -> IdiomClassificationSummary:
    """Classify all images for an idiom.
    
    Args:
        image_data_list: List of ImageData for an idiom
        config: Classification configuration
        
    Returns:
        IdiomClassificationSummary with all classifications
    """
    if not image_data_list:
        raise ValueError("Cannot classify empty image list")
    
    idiom_id = image_data_list[0].annotation.idiom_id
    
    fig_images = []
    lit_images = []
    rand_images = []
    classifications = {}
    
    for image_data in image_data_list:
        image_id = image_data.annotation.image_id
        result = classify_image_data(image_data, config)
        classifications[image_id] = result
        
        if result.image_class == ImageClass.I_FIG:
            fig_images.append(image_id)
        elif result.image_class == ImageClass.I_LIT:
            lit_images.append(image_id)
        else:
            rand_images.append(image_id)
    
    return IdiomClassificationSummary(
        idiom_id=idiom_id,
        total_images=len(image_data_list),
        fig_images=sorted(fig_images),
        lit_images=sorted(lit_images),
        rand_images=sorted(rand_images),
        classifications=classifications,
    )


def classify_annotations_list(
    annotations: List[Annotation],
    config: ClassificationConfig = DEFAULT_CLASSIFICATION_CONFIG
) -> Dict[int, ClassificationResult]:
    """Classify a list of annotations.
    
    Args:
        annotations: List of annotations to classify
        config: Classification configuration
        
    Returns:
        Dictionary mapping image_id to ClassificationResult
    """
    return {
        ann.image_id: classify_annotation(ann, config)
        for ann in annotations
    }


# =============================================================================
# RANKING UTILITIES
# =============================================================================

def get_human_ranking_order() -> List[ImageClass]:
    """Get the human ground truth ranking order.
    
    Returns:
        List of ImageClass in order: [I_FIG, I_LIT, I_RAND]
    """
    return [ImageClass.I_FIG, ImageClass.I_LIT, ImageClass.I_RAND]


def create_human_ranked_list(
    summary: IdiomClassificationSummary
) -> List[Tuple[int, ImageClass]]:
    """Create a human ground truth ranked list for an idiom.
    
    Images are ordered by class (I_FIG first, then I_LIT, then I_RAND),
    with ties broken by image ID.
    
    Args:
        summary: Classification summary for an idiom
        
    Returns:
        List of (image_id, image_class) tuples in ranked order
    """
    ranked = []
    
    # Add I_FIG images first
    for image_id in sorted(summary.fig_images):
        ranked.append((image_id, ImageClass.I_FIG))
    
    # Add I_LIT images second
    for image_id in sorted(summary.lit_images):
        ranked.append((image_id, ImageClass.I_LIT))
    
    # Add I_RAND images last
    for image_id in sorted(summary.rand_images):
        ranked.append((image_id, ImageClass.I_RAND))
    
    return ranked


def sort_by_model_score(
    image_data_list: List[ImageData],
    model_key: str,
    score_field: str = "figurative_score",
    descending: bool = True
) -> List[Tuple[int, float]]:
    """Sort images by model score.
    
    Args:
        image_data_list: List of ImageData
        model_key: Which model's scores to use
        score_field: Which score field to sort by
        descending: If True, higher scores rank first
        
    Returns:
        List of (image_id, score) tuples in ranked order
    """
    scored = []
    for image_data in image_data_list:
        if model_key in image_data.model_outputs:
            output = image_data.model_outputs[model_key]
            score = output.get_score(score_field)
            scored.append((image_data.annotation.image_id, score))
    
    scored.sort(key=lambda x: x[1], reverse=descending)
    return scored


def create_model_ranked_list(
    image_data_list: List[ImageData],
    model_key: str,
    summary: IdiomClassificationSummary,
    score_field: str = "figurative_score"
) -> List[Tuple[int, ImageClass]]:
    """Create a model-based ranked list for comparison with human ranking.
    
    Images are sorted by model score (descending), and each image
    is labeled with its human-assigned class.
    
    Args:
        image_data_list: List of ImageData
        model_key: Which model's scores to use
        summary: Classification summary (for class labels)
        score_field: Which score field to sort by
        
    Returns:
        List of (image_id, image_class) tuples in model-ranked order
    """
    # Sort by model score
    sorted_by_score = sort_by_model_score(
        image_data_list, model_key, score_field, descending=True
    )
    
    # Add class labels
    ranked = []
    for image_id, score in sorted_by_score:
        if image_id in summary.classifications:
            image_class = summary.classifications[image_id].image_class
            ranked.append((image_id, image_class))
    
    return ranked


# =============================================================================
# NUMERICAL SCORING SYSTEM
# =============================================================================

@dataclass
class ScoredImage:
    """An image with its numerical human score.
    
    Attributes:
        image_id: The image identifier
        human_score: Numerical score from annotations (0-100)
        annotations: Original annotation labels
    """
    image_id: int
    human_score: int
    annotations: List[str]


def calculate_annotation_score(
    annotation: Annotation,
    config: ScoringConfig = DEFAULT_SCORING_CONFIG
) -> int:
    """Calculate numerical score for an annotation.
    
    Each of the 5 annotations is weighted according to the scoring config:
    - Figurative+Literal = 20
    - Figurative = 15
    - Literal = 10
    - Partial Literal = 5
    - None = 0
    
    Total score range: 0 (all None) to 100 (all Figurative+Literal)
    
    Args:
        annotation: The annotation to score
        config: Scoring configuration with weights
        
    Returns:
        Total numerical score (sum of all 5 annotation weights)
    """
    return config.calculate_image_score(annotation.annotations)


def calculate_image_data_score(
    image_data: ImageData,
    config: ScoringConfig = DEFAULT_SCORING_CONFIG
) -> int:
    """Calculate numerical score for an ImageData object.
    
    Args:
        image_data: Combined annotation and model output data
        config: Scoring configuration
        
    Returns:
        Total numerical score
    """
    return calculate_annotation_score(image_data.annotation, config)


def score_all_images(
    image_data_list: List[ImageData],
    config: ScoringConfig = DEFAULT_SCORING_CONFIG
) -> List[ScoredImage]:
    """Score all images in a list.
    
    Args:
        image_data_list: List of ImageData objects
        config: Scoring configuration
        
    Returns:
        List of ScoredImage objects
    """
    scored = []
    for image_data in image_data_list:
        ann = image_data.annotation
        score = calculate_annotation_score(ann, config)
        scored.append(ScoredImage(
            image_id=ann.image_id,
            human_score=score,
            annotations=ann.annotations,
        ))
    return scored


@dataclass
class TieGroup:
    """A group of images with the same human score.
    
    Attributes:
        score: The shared human score
        image_ids: List of image IDs with this score
    """
    score: int
    image_ids: List[int]
    
    def __len__(self) -> int:
        return len(self.image_ids)
    
    def is_tie(self) -> bool:
        """Returns True if this group has multiple images (is a tie)."""
        return len(self.image_ids) > 1


@dataclass
class RankedListWithTies:
    """A ranked list with tie group information.
    
    Attributes:
        ranked_image_ids: Image IDs in descending score order
        image_scores: Mapping from image_id to human score
        tie_groups: List of TieGroup objects, ordered by descending score
        
    Note:
        Within each tie group, images are ordered by image_id for consistency,
        but any ordering is considered valid for RBO comparison.
    """
    ranked_image_ids: List[int]
    image_scores: Dict[int, int]
    tie_groups: List[TieGroup]
    
    def get_score(self, image_id: int) -> int:
        """Get the human score for an image."""
        return self.image_scores.get(image_id, 0)
    
    def get_tie_group_for_image(self, image_id: int) -> Optional[TieGroup]:
        """Get the tie group containing an image."""
        for group in self.tie_groups:
            if image_id in group.image_ids:
                return group
        return None
    
    def are_tied(self, image_id_1: int, image_id_2: int) -> bool:
        """Check if two images have the same human score (are tied)."""
        return self.image_scores.get(image_id_1) == self.image_scores.get(image_id_2)


def create_human_ranked_list_by_score(
    image_data_list: List[ImageData],
    config: ScoringConfig = DEFAULT_SCORING_CONFIG
) -> RankedListWithTies:
    """Create a human ground truth ranked list based on numerical scores.
    
    Images are ranked by their summed annotation scores in descending order.
    Ties are tracked for proper RBO comparison.
    
    Args:
        image_data_list: List of ImageData for an idiom
        config: Scoring configuration
        
    Returns:
        RankedListWithTies with ranked images and tie information
    """
    # Score all images
    scored_images = score_all_images(image_data_list, config)
    
    # Build score mapping
    image_scores = {si.image_id: si.human_score for si in scored_images}
    
    # Group by score
    score_to_images: Dict[int, List[int]] = {}
    for si in scored_images:
        if si.human_score not in score_to_images:
            score_to_images[si.human_score] = []
        score_to_images[si.human_score].append(si.image_id)
    
    # Sort scores descending and create tie groups
    sorted_scores = sorted(score_to_images.keys(), reverse=True)
    tie_groups = []
    ranked_image_ids = []
    
    for score in sorted_scores:
        image_ids = sorted(score_to_images[score])  # Sort by ID for consistency
        tie_groups.append(TieGroup(score=score, image_ids=image_ids))
        ranked_image_ids.extend(image_ids)
    
    return RankedListWithTies(
        ranked_image_ids=ranked_image_ids,
        image_scores=image_scores,
        tie_groups=tie_groups,
    )


def create_model_ranked_list_with_scores(
    image_data_list: List[ImageData],
    model_key: str,
    score_field: str = "figurative_score"
) -> Tuple[List[int], Dict[int, float]]:
    """Create a model-based ranked list with score information.
    
    Images are sorted by model score (descending).
    
    Args:
        image_data_list: List of ImageData
        model_key: Which model's scores to use
        score_field: Which score field to sort by
        
    Returns:
        Tuple of (ranked_image_ids, image_id_to_model_score)
    """
    scored = []
    for image_data in image_data_list:
        if model_key in image_data.model_outputs:
            output = image_data.model_outputs[model_key]
            score = output.get_score(score_field)
            scored.append((image_data.annotation.image_id, score))
    
    # Sort by score descending
    scored.sort(key=lambda x: x[1], reverse=True)
    
    ranked_ids = [item[0] for item in scored]
    score_map = {item[0]: item[1] for item in scored}
    
    return ranked_ids, score_map


# =============================================================================
# STATISTICS AND REPORTING
# =============================================================================

def get_classification_statistics(
    summaries: List[IdiomClassificationSummary]
) -> Dict[str, float]:
    """Get aggregate classification statistics across idioms.
    
    Args:
        summaries: List of IdiomClassificationSummary
        
    Returns:
        Dictionary with statistics
    """
    total_images = sum(s.total_images for s in summaries)
    total_fig = sum(s.fig_count for s in summaries)
    total_lit = sum(s.lit_count for s in summaries)
    total_rand = sum(s.rand_count for s in summaries)
    
    return {
        "total_idioms": len(summaries),
        "total_images": total_images,
        "fig_count": total_fig,
        "lit_count": total_lit,
        "rand_count": total_rand,
        "fig_ratio": total_fig / total_images if total_images > 0 else 0.0,
        "lit_ratio": total_lit / total_images if total_images > 0 else 0.0,
        "rand_ratio": total_rand / total_images if total_images > 0 else 0.0,
    }


if __name__ == "__main__":
    # Test classification
    print("=" * 60)
    print("T2IMVI Classification Module - Test")
    print("=" * 60)
    
    from data_loader import load_all_annotations_for_idiom, discover_idioms
    
    idiom_ids = discover_idioms()
    if idiom_ids:
        first_idiom = idiom_ids[0]
        print(f"\nClassifying images for idiom {first_idiom}...")
        
        annotations = load_all_annotations_for_idiom(first_idiom)
        classifications = classify_annotations_list(annotations)
        
        fig_count = sum(1 for r in classifications.values() if r.image_class == ImageClass.I_FIG)
        lit_count = sum(1 for r in classifications.values() if r.image_class == ImageClass.I_LIT)
        rand_count = sum(1 for r in classifications.values() if r.image_class == ImageClass.I_RAND)
        
        print(f"  Total images: {len(annotations)}")
        print(f"  I_fig: {fig_count}")
        print(f"  I_lit: {lit_count}")
        print(f"  I_rand: {rand_count}")
        
        # Show a few examples
        print("\n  Sample classifications:")
        for image_id, result in list(classifications.items())[:5]:
            print(f"    Image {image_id}: {result.image_class.value} ({result.reasoning})")
