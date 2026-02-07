"""
Metrics module for T2IMVI Reliability Analysis.

This module implements the evaluation metrics used in reliability experiments:
- RBO (Rank-Biased Overlap): For Experiment I - Ranking Alignment
- SimPD (Similarity of Promotion and Demotion): For Experiment II - Score Stability

References:
- RBO: Webber et al. (2010) "A Similarity Measure for Indefinite Rankings"
- SimPD: Based on promotion/demotion magnitude analysis for rank volatility
"""

from typing import List, Tuple, Set, Optional, Dict, Any
from dataclasses import dataclass
import math
import logging

logger = logging.getLogger(__name__)


# =============================================================================
# RBO (RANK-BIASED OVERLAP) IMPLEMENTATION
# =============================================================================

def rbo_at_depth(
    list1: List[Any],
    list2: List[Any],
    depth: int
) -> float:
    """Calculate the overlap proportion at a specific depth.
    
    Args:
        list1: First ranked list
        list2: Second ranked list
        depth: Depth to calculate overlap at (1-indexed)
        
    Returns:
        Overlap proportion at depth d: |L1[:d] ∩ L2[:d]| / d
    """
    if depth <= 0:
        return 0.0
    
    set1 = set(list1[:depth])
    set2 = set(list2[:depth])
    
    intersection = len(set1 & set2)
    return intersection / depth


def rbo_min(
    list1: List[Any],
    list2: List[Any],
    p: float = 0.9,
    max_depth: Optional[int] = None
) -> float:
    """Calculate the minimum RBO (lower bound).
    
    This is the minimum possible RBO value given the observed prefix.
    It assumes no further overlap beyond the evaluated depth.
    
    Args:
        list1: First ranked list
        list2: Second ranked list
        p: Persistence parameter (0 < p < 1), higher = more top-weighted
        max_depth: Maximum depth to consider (default: min length of lists)
        
    Returns:
        Minimum RBO value
    """
    if max_depth is None:
        max_depth = min(len(list1), len(list2))
    
    if max_depth == 0:
        return 0.0
    
    rbo_sum = 0.0
    for d in range(1, max_depth + 1):
        overlap = rbo_at_depth(list1, list2, d)
        rbo_sum += (p ** (d - 1)) * overlap
    
    return (1 - p) * rbo_sum


def rbo_ext(
    list1: List[Any],
    list2: List[Any],
    p: float = 0.9
) -> float:
    """Calculate the extrapolated RBO.
    
    This extrapolates the RBO to infinity based on the observed overlap.
    
    RBO(L1, L2, p) = (1-p) * Σ(d=1→∞) p^(d-1) * A_d
    
    Where A_d is the overlap agreement at depth d.
    
    Args:
        list1: First ranked list
        list2: Second ranked list
        p: Persistence parameter (0 < p < 1), higher = more top-weighted
        
    Returns:
        Extrapolated RBO value
    """
    # Handle edge cases
    if not list1 or not list2:
        return 0.0
    
    # Determine the evaluation depth
    k = min(len(list1), len(list2))
    
    # Calculate agreement at each depth
    rbo_sum = 0.0
    x_d = 0  # Running intersection size
    
    set1 = set()
    set2 = set()
    
    for d in range(1, k + 1):
        # Add items at this depth
        if d <= len(list1):
            set1.add(list1[d - 1])
        if d <= len(list2):
            set2.add(list2[d - 1])
        
        # Calculate overlap at this depth
        x_d = len(set1 & set2)
        agreement = x_d / d
        
        rbo_sum += (p ** (d - 1)) * agreement
    
    # Apply the (1-p) factor
    rbo_min_val = (1 - p) * rbo_sum
    
    # Extrapolation: add the expected contribution from beyond depth k
    # RBO_ext = RBO_min + p^k * agreement_at_k
    agreement_at_k = x_d / k if k > 0 else 0
    rbo_ext_val = rbo_min_val + (p ** k) * agreement_at_k
    
    return rbo_ext_val


def rbo(
    list1: List[Any],
    list2: List[Any],
    p: float = 0.9
) -> float:
    """Calculate the Rank-Biased Overlap (RBO) between two ranked lists.
    
    RBO measures the similarity between two ranked lists, with emphasis
    on the top of the lists controlled by parameter p.
    
    Args:
        list1: First ranked list (items in rank order)
        list2: Second ranked list (items in rank order)
        p: Persistence parameter (0 < p < 1)
           - Higher p = more emphasis on top ranks
           - p=0.9 means top item contributes ~10% weight
           
    Returns:
        RBO score in [0, 1], where 1 = identical rankings
        
    Example:
        >>> rbo(['a', 'b', 'c'], ['a', 'b', 'c'], p=0.9)
        1.0
        >>> rbo(['a', 'b', 'c'], ['c', 'b', 'a'], p=0.9)
        0.5  # approximately
    """
    return rbo_ext(list1, list2, p)


# =============================================================================
# RBO RESULT STRUCTURE
# =============================================================================

@dataclass
class RBOResult:
    """Result of RBO calculation.
    
    Attributes:
        score: The RBO score
        p: Persistence parameter used
        list1_length: Length of first list
        list2_length: Length of second list
        overlap_at_depths: Overlap proportions at each depth
    """
    score: float
    p: float
    list1_length: int
    list2_length: int
    overlap_at_depths: List[float]


def calculate_rbo_detailed(
    list1: List[Any],
    list2: List[Any],
    p: float = 0.9
) -> RBOResult:
    """Calculate RBO with detailed results.
    
    Args:
        list1: First ranked list
        list2: Second ranked list
        p: Persistence parameter
        
    Returns:
        RBOResult with score and detailed information
    """
    k = min(len(list1), len(list2))
    overlaps = [rbo_at_depth(list1, list2, d) for d in range(1, k + 1)]
    
    return RBOResult(
        score=rbo(list1, list2, p),
        p=p,
        list1_length=len(list1),
        list2_length=len(list2),
        overlap_at_depths=overlaps,
    )


# =============================================================================
# SIMPD (SIMILARITY OF PROMOTION AND DEMOTION) IMPLEMENTATION
# =============================================================================

def calculate_promotion_magnitude(rank1: int, rank2: int) -> int:
    """Calculate promotion magnitude for an item.
    
    Promotion occurs when an item's rank improves (lower rank number).
    
    Args:
        rank1: Rank in first list (1-indexed)
        rank2: Rank in second list (1-indexed)
        
    Returns:
        Promotion magnitude (0 if not promoted)
    """
    if rank1 > rank2:
        return rank1 - rank2
    return 0


def calculate_demotion_magnitude(rank1: int, rank2: int) -> int:
    """Calculate demotion magnitude for an item.
    
    Demotion occurs when an item's rank worsens (higher rank number).
    
    Args:
        rank1: Rank in first list (1-indexed)
        rank2: Rank in second list (1-indexed)
        
    Returns:
        Demotion magnitude (0 if not demoted)
    """
    if rank2 > rank1:
        return rank2 - rank1
    return 0


def calculate_normalization_factor(m: int) -> float:
    """Calculate the normalization factor N(M).
    
    N(M) = (M+1)^2 / 4 if M is odd
    N(M) = M(M+2) / 4 if M is even
    
    Args:
        m: List length minus 1
        
    Returns:
        Normalization factor
    """
    if m <= 0:
        return 1.0  # Avoid division by zero
    
    if m % 2 == 1:  # odd
        return ((m + 1) ** 2) / 4
    else:  # even
        return (m * (m + 2)) / 4


def get_item_ranks(ranked_list: List[Any]) -> Dict[Any, int]:
    """Get a mapping from items to their ranks (1-indexed).
    
    Args:
        ranked_list: List of items in rank order
        
    Returns:
        Dictionary mapping item to rank (1-indexed)
    """
    return {item: rank + 1 for rank, item in enumerate(ranked_list)}


def simpd_base(
    list1: List[Any],
    list2: List[Any]
) -> float:
    """Calculate base SimPD similarity.
    
    SimPD measures similarity based on promotion and demotion magnitudes.
    
    Formula:
    - SP = Σ P(j) for all overlapping items
    - SD = Σ D(j) for all overlapping items
    - NP = SP / N(l-1), ND = SD / N(l'-1)
    - SimP = 1 - NP, SimD = 1 - ND
    - SimPD = (SimP + SimD) / 2
    
    Args:
        list1: First ranked list
        list2: Second ranked list
        
    Returns:
        SimPD score in [0, 1], where 1 = identical rankings
    """
    if not list1 or not list2:
        return 1.0 if list1 == list2 else 0.0
    
    # Get ranks for each list
    ranks1 = get_item_ranks(list1)
    ranks2 = get_item_ranks(list2)
    
    # Find overlapping items
    overlap = set(ranks1.keys()) & set(ranks2.keys())
    
    if not overlap:
        return 0.0
    
    # Calculate promotion and demotion sums
    sp = 0  # Sum of promotions
    sd = 0  # Sum of demotions
    
    for item in overlap:
        r1 = ranks1[item]
        r2 = ranks2[item]
        sp += calculate_promotion_magnitude(r1, r2)
        sd += calculate_demotion_magnitude(r1, r2)
    
    # Calculate normalization factors
    n_promotion = calculate_normalization_factor(len(list1) - 1)
    n_demotion = calculate_normalization_factor(len(list2) - 1)
    
    # Normalize
    np = sp / n_promotion if n_promotion > 0 else 0
    nd = sd / n_demotion if n_demotion > 0 else 0
    
    # Calculate similarities
    sim_p = 1 - np
    sim_d = 1 - nd
    
    # Final SimPD
    simpd = (sim_p + sim_d) / 2
    
    # Clamp to [0, 1] in case of numerical issues
    return max(0.0, min(1.0, simpd))


def simpd_f(
    list1: List[Any],
    list2: List[Any],
    focus_items: Optional[Set[Any]] = None
) -> float:
    """Calculate SimPD-F (Focussing on Flagged Items).
    
    Only considers rank changes for a specific subset of items.
    
    Args:
        list1: First ranked list
        list2: Second ranked list
        focus_items: Set of items to focus on (if None, uses all overlapping)
        
    Returns:
        SimPD-F score in [0, 1]
    """
    if not list1 or not list2:
        return 1.0 if list1 == list2 else 0.0
    
    ranks1 = get_item_ranks(list1)
    ranks2 = get_item_ranks(list2)
    
    # Find overlapping items
    overlap = set(ranks1.keys()) & set(ranks2.keys())
    
    # Filter to focus items if specified
    if focus_items is not None:
        overlap = overlap & focus_items
    
    if not overlap:
        return 0.0
    
    # Calculate promotion and demotion sums for focus items only
    sp = 0
    sd = 0
    
    for item in overlap:
        r1 = ranks1[item]
        r2 = ranks2[item]
        sp += calculate_promotion_magnitude(r1, r2)
        sd += calculate_demotion_magnitude(r1, r2)
    
    n_promotion = calculate_normalization_factor(len(list1) - 1)
    n_demotion = calculate_normalization_factor(len(list2) - 1)
    
    np = sp / n_promotion if n_promotion > 0 else 0
    nd = sd / n_demotion if n_demotion > 0 else 0
    
    sim_p = 1 - np
    sim_d = 1 - nd
    
    return max(0.0, min(1.0, (sim_p + sim_d) / 2))


def f_measure(m1: float, m2: float, beta: float = 1.0) -> float:
    """Calculate F-measure combining two metrics.
    
    F(M1, M2, β) = (β² + 1) * M1 * M2 / (β² * M1 + M2)
    
    Args:
        m1: First metric value
        m2: Second metric value
        beta: Beta parameter (default 1.0 for balanced F1)
        
    Returns:
        F-measure value
    """
    if m1 <= 0 or m2 <= 0:
        return 0.0
    
    numerator = (beta ** 2 + 1) * m1 * m2
    denominator = (beta ** 2) * m1 + m2
    
    if denominator == 0:
        return 0.0
    
    return numerator / denominator


def agreement(list1: List[Any], list2: List[Any]) -> float:
    """Calculate Jaccard-like agreement between two lists.
    
    Agree(L1, L2) = |S(L1) ∩ S(L2)| / |S(L1) ∪ S(L2)|
    
    Args:
        list1: First list
        list2: Second list
        
    Returns:
        Agreement value in [0, 1]
    """
    set1 = set(list1)
    set2 = set(list2)
    
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    
    if union == 0:
        return 1.0
    
    return intersection / union


def simpd_a(
    list1: List[Any],
    list2: List[Any],
    beta: float = 1.0
) -> float:
    """Calculate SimPD-A (Penalizing Dangling Items).
    
    Combines SimPD with item agreement using F-measure.
    
    SimPD-A = F(Agree(L1, L2), SimPD(L1, L2), β)
    
    Args:
        list1: First ranked list
        list2: Second ranked list
        beta: Beta for F-measure
        
    Returns:
        SimPD-A score in [0, 1]
    """
    agree = agreement(list1, list2)
    simpd = simpd_base(list1, list2)
    
    return f_measure(agree, simpd, beta)


def top_agreement(
    list1: List[Any],
    list2: List[Any],
    max_depth: Optional[int] = None
) -> float:
    """Calculate top-heavy agreement.
    
    tAgree(L1, L2) = (1/M) * Σ(r=1→M) Agree(L1@r, L2@r)
    
    Args:
        list1: First ranked list
        list2: Second ranked list
        max_depth: Maximum depth M (default: max length of lists)
        
    Returns:
        Top-heavy agreement value in [0, 1]
    """
    if max_depth is None:
        max_depth = max(len(list1), len(list2))
    
    if max_depth == 0:
        return 1.0
    
    total_agreement = 0.0
    for r in range(1, max_depth + 1):
        total_agreement += agreement(list1[:r], list2[:r])
    
    return total_agreement / max_depth


def simpd_ta(
    list1: List[Any],
    list2: List[Any],
    beta: float = 1.0
) -> float:
    """Calculate SimPD-tA (Top-heavy Agreement).
    
    SimPD-tA = F(tAgree(L1, L2), SimPD(L1, L2), β)
    
    Args:
        list1: First ranked list
        list2: Second ranked list
        beta: Beta for F-measure
        
    Returns:
        SimPD-tA score in [0, 1]
    """
    t_agree = top_agreement(list1, list2)
    simpd = simpd_base(list1, list2)
    
    return f_measure(t_agree, simpd, beta)


def simpd(
    list1: List[Any],
    list2: List[Any],
    variant: str = "base",
    beta: float = 1.0,
    focus_items: Optional[Set[Any]] = None
) -> float:
    """Calculate SimPD with specified variant.
    
    Args:
        list1: First ranked list
        list2: Second ranked list
        variant: Which variant to use ('base', 'F', 'A', 'tA')
        beta: Beta parameter for F-measure variants
        focus_items: Focus items for SimPD-F variant
        
    Returns:
        SimPD score in [0, 1]
    """
    if variant == "base":
        return simpd_base(list1, list2)
    elif variant == "F":
        return simpd_f(list1, list2, focus_items)
    elif variant == "A":
        return simpd_a(list1, list2, beta)
    elif variant == "tA":
        return simpd_ta(list1, list2, beta)
    else:
        raise ValueError(f"Unknown SimPD variant: {variant}")


# =============================================================================
# SIMPD RESULT STRUCTURE
# =============================================================================

@dataclass
class SimPDResult:
    """Result of SimPD calculation.
    
    Attributes:
        score: The SimPD score
        variant: Which variant was used
        sum_promotions: Total promotion magnitude
        sum_demotions: Total demotion magnitude
        sim_p: Promotion similarity component
        sim_d: Demotion similarity component
        overlap_count: Number of overlapping items
    """
    score: float
    variant: str
    sum_promotions: int
    sum_demotions: int
    sim_p: float
    sim_d: float
    overlap_count: int


def calculate_simpd_detailed(
    list1: List[Any],
    list2: List[Any],
    variant: str = "base",
    beta: float = 1.0
) -> SimPDResult:
    """Calculate SimPD with detailed results.
    
    Args:
        list1: First ranked list
        list2: Second ranked list
        variant: Which variant to use
        beta: Beta parameter
        
    Returns:
        SimPDResult with score and detailed information
    """
    ranks1 = get_item_ranks(list1)
    ranks2 = get_item_ranks(list2)
    overlap = set(ranks1.keys()) & set(ranks2.keys())
    
    sp = 0
    sd = 0
    for item in overlap:
        r1 = ranks1[item]
        r2 = ranks2[item]
        sp += calculate_promotion_magnitude(r1, r2)
        sd += calculate_demotion_magnitude(r1, r2)
    
    n_promotion = calculate_normalization_factor(len(list1) - 1)
    n_demotion = calculate_normalization_factor(len(list2) - 1)
    
    np = sp / n_promotion if n_promotion > 0 else 0
    nd = sd / n_demotion if n_demotion > 0 else 0
    
    sim_p = 1 - np
    sim_d = 1 - nd
    
    return SimPDResult(
        score=simpd(list1, list2, variant, beta),
        variant=variant,
        sum_promotions=sp,
        sum_demotions=sd,
        sim_p=sim_p,
        sim_d=sim_d,
        overlap_count=len(overlap),
    )


# =============================================================================
# NEW METRICS: ICC, PEARSON, MAE (for Experiment I - Ranking Alignment)
# =============================================================================

def mean(values: List[float]) -> float:
    """Calculate arithmetic mean.
    
    Args:
        values: List of numerical values
        
    Returns:
        Arithmetic mean
    """
    if not values:
        return 0.0
    return sum(values) / len(values)


def variance(values: List[float], ddof: int = 0) -> float:
    """Calculate variance.
    
    Args:
        values: List of numerical values
        ddof: Delta degrees of freedom (0 for population, 1 for sample)
        
    Returns:
        Variance
    """
    n = len(values)
    if n <= ddof:
        return 0.0
    
    m = mean(values)
    return sum((x - m) ** 2 for x in values) / (n - ddof)


def std(values: List[float], ddof: int = 0) -> float:
    """Calculate standard deviation.
    
    Args:
        values: List of numerical values
        ddof: Delta degrees of freedom
        
    Returns:
        Standard deviation
    """
    return math.sqrt(variance(values, ddof))


def covariance(x: List[float], y: List[float], ddof: int = 0) -> float:
    """Calculate covariance between two variables.
    
    Args:
        x: First variable values
        y: Second variable values
        ddof: Delta degrees of freedom
        
    Returns:
        Covariance
    """
    n = len(x)
    if n != len(y) or n <= ddof:
        return 0.0
    
    mean_x = mean(x)
    mean_y = mean(y)
    
    return sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y)) / (n - ddof)


def pearson_correlation(
    x: List[float],
    y: List[float]
) -> float:
    """Calculate Pearson correlation coefficient.
    
    Measures linear correlation between two variables.
    
    Formula: r = cov(X,Y) / (std(X) * std(Y))
    
    Args:
        x: First variable values (e.g., human scores)
        y: Second variable values (e.g., model scores)
        
    Returns:
        Pearson r in [-1, 1], where:
        - 1 = perfect positive correlation
        - 0 = no linear correlation
        - -1 = perfect negative correlation
    """
    if len(x) != len(y) or len(x) < 2:
        return 0.0
    
    cov = covariance(x, y, ddof=0)
    std_x = std(x, ddof=0)
    std_y = std(y, ddof=0)
    
    if std_x == 0 or std_y == 0:
        return 0.0
    
    r = cov / (std_x * std_y)
    
    # Clamp to [-1, 1] for numerical stability
    return max(-1.0, min(1.0, r))


def mae(
    x: List[float],
    y: List[float]
) -> float:
    """Calculate Mean Absolute Error.
    
    MAE = (1/n) * Σ|x_i - y_i|
    
    Args:
        x: First variable values (e.g., human scores)
        y: Second variable values (e.g., model scores)
        
    Returns:
        MAE (non-negative), lower is better
    """
    if len(x) != len(y) or not x:
        return 0.0
    
    return sum(abs(xi - yi) for xi, yi in zip(x, y)) / len(x)


def normalized_mae(
    x: List[float],
    y: List[float],
    max_possible_error: Optional[float] = None
) -> float:
    """Calculate normalized MAE (0-1 range).
    
    Useful when comparing scores on different scales.
    
    Args:
        x: First variable values
        y: Second variable values
        max_possible_error: Maximum possible error for normalization
                           (default: max(x) - min(x) or max(y) - min(y))
        
    Returns:
        Normalized MAE in [0, 1], lower is better
    """
    if not x or not y:
        return 0.0
    
    raw_mae = mae(x, y)
    
    if max_possible_error is None:
        # Estimate from data range
        all_values = list(x) + list(y)
        max_possible_error = max(all_values) - min(all_values)
    
    if max_possible_error == 0:
        return 0.0
    
    return raw_mae / max_possible_error


def icc(
    ratings: List[List[float]],
    icc_type: str = "ICC(2,1)"
) -> float:
    """Calculate Intraclass Correlation Coefficient.
    
    ICC measures the reliability/agreement of ratings by comparing
    variance between subjects to variance within subjects.
    
    For our use case (human scores vs model scores):
    - Each "subject" is an image
    - Each "rater" is a scoring method (human vs model)
    
    Formula (ICC(2,1) - Two-way random, single measurement):
    ICC = (MS_between - MS_error) / (MS_between + (k-1)*MS_error + k*(MS_rater - MS_error)/n)
    
    Simplified for 2 raters (human and model):
    ICC = (MS_between - MS_within) / (MS_between + MS_within)
    
    Args:
        ratings: List of [human_score, model_score] pairs for each subject
                 Shape: n_subjects x 2
        icc_type: Type of ICC (currently supports "ICC(2,1)" and "ICC(1,1)")
        
    Returns:
        ICC value in [-1, 1] typically, where:
        - 1 = perfect agreement
        - 0 = no agreement beyond chance
        - <0 = systematic disagreement
    """
    n = len(ratings)
    if n < 2:
        return 0.0
    
    # Number of raters (for our case, always 2: human and model)
    k = len(ratings[0]) if ratings else 0
    if k < 2:
        return 0.0
    
    # Calculate grand mean
    all_values = [r for row in ratings for r in row]
    grand_mean = mean(all_values)
    
    # Calculate row means (subject means)
    row_means = [mean(row) for row in ratings]
    
    # Calculate column means (rater means)
    col_means = [mean([ratings[i][j] for i in range(n)]) for j in range(k)]
    
    # Calculate Sum of Squares
    # SS_total: total variance
    ss_total = sum((x - grand_mean) ** 2 for x in all_values)
    
    # SS_between (between subjects): variance due to differences between subjects
    ss_between = k * sum((rm - grand_mean) ** 2 for rm in row_means)
    
    # SS_raters (between raters): variance due to differences between raters
    ss_raters = n * sum((cm - grand_mean) ** 2 for cm in col_means)
    
    # SS_error (residual): unexplained variance
    ss_error = ss_total - ss_between - ss_raters
    
    # Calculate Mean Squares
    df_between = n - 1
    df_raters = k - 1
    df_error = (n - 1) * (k - 1)
    
    ms_between = ss_between / df_between if df_between > 0 else 0
    ms_raters = ss_raters / df_raters if df_raters > 0 else 0
    ms_error = ss_error / df_error if df_error > 0 else 0
    
    # Calculate ICC based on type
    if icc_type == "ICC(1,1)":
        # One-way random effects
        # ICC(1,1) = (MS_between - MS_within) / (MS_between + (k-1)*MS_within)
        ms_within = (ss_raters + ss_error) / (n * (k - 1)) if n * (k - 1) > 0 else 0
        denominator = ms_between + (k - 1) * ms_within
        if denominator == 0:
            return 0.0
        return (ms_between - ms_within) / denominator
    
    elif icc_type == "ICC(2,1)":
        # Two-way random effects, single measurement
        # ICC(2,1) = (MS_between - MS_error) / (MS_between + (k-1)*MS_error + k*(MS_raters - MS_error)/n)
        denominator = ms_between + (k - 1) * ms_error + k * (ms_raters - ms_error) / n
        if denominator == 0:
            return 0.0
        return (ms_between - ms_error) / denominator
    
    elif icc_type == "ICC(3,1)":
        # Two-way mixed effects, single measurement
        # ICC(3,1) = (MS_between - MS_error) / (MS_between + (k-1)*MS_error)
        denominator = ms_between + (k - 1) * ms_error
        if denominator == 0:
            return 0.0
        return (ms_between - ms_error) / denominator
    
    else:
        raise ValueError(f"Unknown ICC type: {icc_type}")


def icc_from_pairs(
    x: List[float],
    y: List[float],
    icc_type: str = "ICC(2,1)"
) -> float:
    """Calculate ICC from two lists of paired values.
    
    Convenience function for our use case where we have:
    - x = human scores
    - y = model scores (normalized to same scale)
    
    Args:
        x: First rater values (human scores)
        y: Second rater values (model scores)
        icc_type: Type of ICC to calculate
        
    Returns:
        ICC value
    """
    if len(x) != len(y):
        raise ValueError("Lists must have equal length")
    
    ratings = [[xi, yi] for xi, yi in zip(x, y)]
    return icc(ratings, icc_type)


# =============================================================================
# RBO WITH TIE HANDLING
# =============================================================================

def rbo_with_ties(
    human_ranking: List[int],
    model_ranking: List[int],
    tie_groups: List[Tuple[int, List[int]]],
    p: float = 0.9
) -> float:
    """Calculate RBO with tie handling for human scores.
    
    When human scores are tied, any ordering of tied items in the model
    ranking is considered equally valid. We compute the RBO for the
    best possible ordering of tied items.
    
    Algorithm:
    1. For each depth d, calculate the overlap
    2. When checking overlap, items in the same tie group are considered
       interchangeable - we give credit if the model rank contains ANY
       item from the same tie group
    
    Args:
        human_ranking: Image IDs in human score order (descending)
        model_ranking: Image IDs in model score order (descending)
        tie_groups: List of (score, [image_ids]) tuples representing ties
        p: RBO persistence parameter
        
    Returns:
        RBO score accounting for ties
    """
    if not human_ranking or not model_ranking:
        return 0.0
    
    # Build a mapping from image_id to its tie group
    image_to_tie_group: Dict[int, Set[int]] = {}
    for score, image_ids in tie_groups:
        group_set = set(image_ids)
        for img_id in image_ids:
            image_to_tie_group[img_id] = group_set
    
    # For single images (no tie), they are their own group
    for img_id in human_ranking:
        if img_id not in image_to_tie_group:
            image_to_tie_group[img_id] = {img_id}
    
    k = min(len(human_ranking), len(model_ranking))
    
    rbo_sum = 0.0
    
    for d in range(1, k + 1):
        # Get top-d items from each list
        human_top_d = human_ranking[:d]
        model_top_d = set(model_ranking[:d])
        
        # Count matches, considering tie groups
        matches = 0
        used_model_items = set()
        
        for h_img in human_top_d:
            tie_group = image_to_tie_group.get(h_img, {h_img})
            
            # Check if any item from the tie group is in model's top-d
            # and hasn't been used yet
            for tied_img in tie_group:
                if tied_img in model_top_d and tied_img not in used_model_items:
                    matches += 1
                    used_model_items.add(tied_img)
                    break
        
        overlap = matches / d
        rbo_sum += (p ** (d - 1)) * overlap
    
    rbo_min_val = (1 - p) * rbo_sum
    
    # Extrapolation
    agreement_at_k = matches / k if k > 0 else 0
    rbo_ext_val = rbo_min_val + (p ** k) * agreement_at_k
    
    return rbo_ext_val


@dataclass
class RBOWithTiesResult:
    """Result of RBO calculation with tie handling.
    
    Attributes:
        score: The RBO score (tie-aware)
        score_without_ties: Standard RBO for comparison
        p: Persistence parameter used
        num_tie_groups: Number of tie groups in human ranking
        max_tie_group_size: Size of largest tie group
    """
    score: float
    score_without_ties: float
    p: float
    num_tie_groups: int
    max_tie_group_size: int


def calculate_rbo_with_ties_detailed(
    human_ranking: List[int],
    model_ranking: List[int],
    tie_groups: List[Tuple[int, List[int]]],
    p: float = 0.9
) -> RBOWithTiesResult:
    """Calculate RBO with ties and provide detailed results.
    
    Args:
        human_ranking: Image IDs in human score order
        model_ranking: Image IDs in model score order
        tie_groups: List of (score, [image_ids]) tuples
        p: RBO persistence parameter
        
    Returns:
        RBOWithTiesResult with detailed information
    """
    score_with_ties = rbo_with_ties(human_ranking, model_ranking, tie_groups, p)
    score_without_ties = rbo(human_ranking, model_ranking, p)
    
    # Count actual tie groups (groups with more than 1 item)
    actual_tie_groups = [g for s, g in tie_groups if len(g) > 1]
    max_size = max((len(g) for s, g in tie_groups), default=0)
    
    return RBOWithTiesResult(
        score=score_with_ties,
        score_without_ties=score_without_ties,
        p=p,
        num_tie_groups=len(actual_tie_groups),
        max_tie_group_size=max_size,
    )


# =============================================================================
# COMPREHENSIVE METRICS RESULT
# =============================================================================

@dataclass
class ExperimentIMetrics:
    """All metrics for Experiment I (Ranking Alignment).
    
    Attributes:
        rbo: Standard RBO score
        rbo_with_ties: RBO score accounting for ties
        icc: Intraclass Correlation Coefficient
        pearson_r: Pearson correlation coefficient
        mae: Mean Absolute Error (raw)
        normalized_mae: Normalized MAE (0-1, lower is better)
        n_images: Number of images evaluated
        n_tie_groups: Number of tie groups in human ranking
    """
    rbo: float
    rbo_with_ties: float
    icc: float
    pearson_r: float
    mae: float
    normalized_mae: float
    n_images: int
    n_tie_groups: int


def calculate_experiment_i_metrics(
    human_scores: List[float],
    model_scores: List[float],
    human_ranking: List[int],
    model_ranking: List[int],
    tie_groups: List[Tuple[int, List[int]]],
    p: float = 0.9,
    human_score_max: float = 100.0
) -> ExperimentIMetrics:
    """Calculate all Experiment I metrics.
    
    Args:
        human_scores: List of human scores (aligned with model_scores)
        model_scores: List of model scores (normalized, aligned with human_scores)
        human_ranking: Image IDs in human ranking order
        model_ranking: Image IDs in model ranking order
        tie_groups: Tie group information from human ranking
        p: RBO persistence parameter
        human_score_max: Maximum possible human score (for MAE normalization)
        
    Returns:
        ExperimentIMetrics with all calculated metrics
    """
    # RBO calculations
    rbo_standard = rbo(human_ranking, model_ranking, p)
    rbo_ties = rbo_with_ties(human_ranking, model_ranking, tie_groups, p)
    
    # ICC calculation
    icc_value = icc_from_pairs(human_scores, model_scores, "ICC(2,1)")
    
    # Pearson correlation
    pearson_value = pearson_correlation(human_scores, model_scores)
    
    # MAE calculations
    mae_raw = mae(human_scores, model_scores)
    mae_norm = normalized_mae(human_scores, model_scores, human_score_max)
    
    # Count actual tie groups
    n_tie_groups = sum(1 for _, imgs in tie_groups if len(imgs) > 1)
    
    return ExperimentIMetrics(
        rbo=rbo_standard,
        rbo_with_ties=rbo_ties,
        icc=icc_value,
        pearson_r=pearson_value,
        mae=mae_raw,
        normalized_mae=mae_norm,
        n_images=len(human_scores),
        n_tie_groups=n_tie_groups,
    )


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def extract_ranking_order(
    ranked_list: List[Tuple[Any, Any]]
) -> List[Any]:
    """Extract just the item identifiers from a ranked list of tuples.
    
    Args:
        ranked_list: List of (item_id, label) tuples in rank order
        
    Returns:
        List of item identifiers only
    """
    return [item_id for item_id, _ in ranked_list]


def extract_class_sequence(
    ranked_list: List[Tuple[Any, Any]]
) -> List[Any]:
    """Extract the class sequence from a ranked list.
    
    Args:
        ranked_list: List of (item_id, class_label) tuples in rank order
        
    Returns:
        List of class labels in order
    """
    return [class_label for _, class_label in ranked_list]


if __name__ == "__main__":
    # Test metrics
    print("=" * 60)
    print("T2IMVI Metrics Module - Test")
    print("=" * 60)
    
    # Test RBO
    print("\n--- RBO Tests ---")
    list_a = ['a', 'b', 'c', 'd', 'e']
    list_b = ['a', 'b', 'c', 'd', 'e']
    print(f"Identical lists: RBO = {rbo(list_a, list_b):.4f}")
    
    list_c = ['e', 'd', 'c', 'b', 'a']
    print(f"Reversed lists: RBO = {rbo(list_a, list_c):.4f}")
    
    list_d = ['a', 'b', 'c', 'x', 'y']
    print(f"Partial overlap: RBO = {rbo(list_a, list_d):.4f}")
    
    # Test SimPD
    print("\n--- SimPD Tests ---")
    print(f"Identical lists: SimPD = {simpd_base(list_a, list_b):.4f}")
    print(f"Reversed lists: SimPD = {simpd_base(list_a, list_c):.4f}")
    print(f"Partial overlap: SimPD = {simpd_base(list_a, list_d):.4f}")
    
    # Test detailed results
    print("\n--- Detailed RBO ---")
    result = calculate_rbo_detailed(list_a, list_d)
    print(f"Score: {result.score:.4f}")
    print(f"Overlaps at depths: {result.overlap_at_depths}")
    
    print("\n--- Detailed SimPD ---")
    result = calculate_simpd_detailed(list_a, list_c)
    print(f"Score: {result.score:.4f}")
    print(f"Sum promotions: {result.sum_promotions}")
    print(f"Sum demotions: {result.sum_demotions}")
