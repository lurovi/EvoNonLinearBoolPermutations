import numpy as np
from generator import Individual
from walsh_transform import WalshTransform


def absolute_difference_between_numbers(x: Individual, y: Individual) -> float:
    """Compute the absolute difference between two numbers.

    Args:
        x (Individual): First number.
        y (Individual): Second number.

    Returns:
        float: Absolute difference between x and y.
    """
    return abs(x.fitness - y.fitness)


def inv_absolute_difference_between_numbers(x: Individual, y: Individual) -> float:
    """Compute the inverse absolute difference between two numbers.

    Args:
        x (Individual): First number.
        y (Individual): Second number.

    Returns:
        float: Inverse absolute difference between x and y.
    """
    return  1 / (1 + abs(x.fitness - y.fitness))


def hamming_distance(x: Individual, y: Individual) -> int:
    """Compute the Hamming distance between two binary vectors.

    Args:
        x (Individual): First binary vector.
        y (Individual): Second binary vector.

    Returns:
        int: Hamming distance between x and y.
    """
    if x.genome.shape != y.genome.shape:
        raise ValueError("Input vectors must have the same shape.")
    return np.sum(x.genome != y.genome)


def euclidean_distance(x: Individual, y: Individual) -> float:
    """Compute the Euclidean distance between two vectors.

    Args:
        x (Individual): First vector.
        y (Individual): Second vector.

    Returns:
        float: Euclidean distance between x and y.
    """
    if x.genome.shape != y.genome.shape:
        raise ValueError("Input vectors must have the same shape.")
    return np.sqrt(np.sum((x.genome - y.genome) ** 2))


def walsh_spectral_distance(x: Individual, y: Individual, walsh: WalshTransform) -> float:
    """Compute the Walsh spectral distance between two vectors.

    Args:
        x (Individual): First vector.
        y (Individual): Second vector.

    Returns:
        float: Walsh spectral distance between x and y.
    """
    if x.genome.shape != y.genome.shape:
        raise ValueError("Input vectors must have the same shape.")

    x_spectrum, _ = walsh.apply(x.genome)
    y_spectrum, _ = walsh.apply(y.genome)
    return np.sqrt(np.sum((x_spectrum - y_spectrum) ** 2))


def abs_walsh_spectral_distance(x: Individual, y: Individual, walsh: WalshTransform) -> float:
    """Compute the absolute Walsh spectral distance between two vectors.

    Args:
        x (Individual): First vector.
        y (Individual): Second vector.

    Returns:
        float: Absolute Walsh spectral distance between x and y.
    """
    if x.genome.shape != y.genome.shape:
        raise ValueError("Input vectors must have the same shape.")

    x_spectrum, _ = walsh.apply(x.genome)
    y_spectrum, _ = walsh.apply(y.genome)
    return np.sqrt(np.sum((np.abs(x_spectrum) - np.abs(y_spectrum)) ** 2))


def walsh_spectral_similarity(x: Individual, y: Individual, walsh: WalshTransform) -> float:
    """Compute the Walsh spectral similarity between two vectors.

    Args:
        x (Individual): First vector.
        y (Individual): Second vector.

    Returns:
        float: Walsh spectral similarity between x and y.
    """
    if x.genome.shape != y.genome.shape:
        raise ValueError("Input vectors must have the same shape.")

    x_spectrum, _ = walsh.apply(x.genome)
    y_spectrum, _ = walsh.apply(y.genome)
    distance = np.sqrt(np.sum((x_spectrum - y_spectrum) ** 2))
    return 1 / (1 + distance)


def abs_walsh_spectral_similarity(x: Individual, y: Individual, walsh: WalshTransform) -> float:
    """Compute the absolute Walsh spectral similarity between two vectors.

    Args:
        x (Individual): First vector.
        y (Individual): Second vector.

    Returns:
        float: Absolute Walsh spectral similarity between x and y.
    """
    if x.genome.shape != y.genome.shape:
        raise ValueError("Input vectors must have the same shape.")

    x_spectrum, _ = walsh.apply(x.genome)
    y_spectrum, _ = walsh.apply(y.genome)
    distance = np.sqrt(np.sum((np.abs(x_spectrum) - np.abs(y_spectrum)) ** 2))
    return 1 / (1 + distance)


def manhattan_distance(x: Individual, y: Individual) -> float:
    """Compute the Manhattan distance between two vectors.

    Args:
        x (Individual): First vector.
        y (Individual): Second vector.

    Returns:
        float: Manhattan distance between x and y.
    """
    if x.genome.shape != y.genome.shape:
        raise ValueError("Input vectors must have the same shape.")
    return np.sum(np.abs(x.genome - y.genome))


def chebyshev_distance(x: Individual, y: Individual) -> float:
    """Compute the Chebyshev distance between two vectors.

    Args:
        x (Individual): First vector.
        y (Individual): Second vector.

    Returns:
        float: Chebyshev distance between x and y.
    """
    if x.genome.shape != y.genome.shape:
        raise ValueError("Input vectors must have the same shape.")
    return np.max(np.abs(x.genome - y.genome))


def minkowski_distance(x: Individual, y: Individual, p: float) -> float:
    """Compute the Minkowski distance between two vectors.

    Args:
        x (Individual): First vector.
        y (Individual): Second vector.
        p (float): Order of the norm.

    Returns:
        float: Minkowski distance between x and y.
    """
    if x.genome.shape != y.genome.shape:
        raise ValueError("Input vectors must have the same shape.")
    if p <= 0:
        raise ValueError("Order p must be a positive number.")
    return np.sum(np.abs(x.genome - y.genome) ** p) ** (1 / p)


def cosine_distance(x: Individual, y: Individual) -> float:
    """Compute the Cosine distance between two vectors.

    Args:
        x (Individual): First vector.
        y (Individual): Second vector.

    Returns:
        float: Cosine distance between x and y.
    """
    if x.genome.shape != y.genome.shape:
        raise ValueError("Input vectors must have the same shape.")
    dot_product = np.dot(x.genome, y.genome)
    norm_x = np.linalg.norm(x.genome)
    norm_y = np.linalg.norm(y.genome)
    if norm_x == 0 or norm_y == 0:
        raise ValueError("Input vectors must not be zero vectors.")
    cosine_similarity = dot_product / (norm_x * norm_y)
    return 1 - cosine_similarity

def jaccard_distance(x: Individual, y: Individual) -> float:
    """Compute the Jaccard distance between two binary vectors.

    Args:
        x (Individual): First binary vector.
        y (Individual): Second binary vector.

    Returns:
        float: Jaccard distance between x and y.
    """
    if x.genome.shape != y.genome.shape:
        raise ValueError("Input vectors must have the same shape.")
    intersection = np.sum(np.logical_and(x.genome, y.genome))
    union = np.sum(np.logical_or(x.genome, y.genome))
    if union == 0:
        return 0.0
    return 1 - intersection / union


def hamming_similarity(x: Individual, y: Individual) -> float:
    """Compute the Hamming similarity between two binary vectors.

    Args:
        x (Individual): First binary vector.
        y (Individual): Second binary vector.

    Returns:
        float: Hamming similarity between x and y.
    """
    if x.genome.shape != y.genome.shape:
        raise ValueError("Input vectors must have the same shape.")
    return np.sum(x.genome == y.genome) / x.genome.size


def jaccard_similarity(x: Individual, y: Individual) -> float:
    """Compute the Jaccard similarity between two binary vectors.

    Args:
        x (Individual): First binary vector.
        y (Individual): Second binary vector.

    Returns:
        float: Jaccard similarity between x and y.
    """
    if x.genome.shape != y.genome.shape:
        raise ValueError("Input vectors must have the same shape.")
    intersection = np.sum(np.logical_and(x.genome, y.genome))
    union = np.sum(np.logical_or(x.genome, y.genome))
    if union == 0:
        return 1.0
    return intersection / union


def pearson_correlation(x: Individual, y: Individual) -> float:
    """Compute the Pearson correlation coefficient between two vectors.

    Args:
        x (Individual): First vector.
        y (Individual): Second vector.

    Returns:
        float: Pearson correlation coefficient between x and y.
    """
    if x.genome.shape != y.genome.shape:
        raise ValueError("Input vectors must have the same shape.")
    if x.genome.size == 0:
        raise ValueError("Input vectors must not be empty.")
    mean_x = np.mean(x.genome)
    mean_y = np.mean(y.genome)
    numerator = np.sum((x.genome - mean_x) * (y.genome - mean_y))
    denominator = np.sqrt(np.sum((x.genome - mean_x) ** 2) * np.sum((y.genome - mean_y) ** 2))
    if denominator == 0:
        return 0.0
    return numerator / denominator


def spearman_correlation(x: Individual, y: Individual) -> float:
    """Compute the Spearman rank correlation coefficient between two vectors.

    Args:
        x (Individual): First vector.
        y (Individual): Second vector.

    Returns:
        float: Spearman rank correlation coefficient between x and y.
    """
    if x.genome.shape != y.genome.shape:
        raise ValueError("Input vectors must have the same shape.")
    if x.genome.size == 0:
        raise ValueError("Input vectors must not be empty.")
    rank_x = np.argsort(np.argsort(x.genome))
    rank_y = np.argsort(np.argsort(y.genome))
    return pearson_correlation(rank_x, rank_y)


def chebyshev_similarity(x: Individual, y: Individual) -> float:
    """Compute the Chebyshev similarity between two vectors.

    Args:
        x (Individual): First vector.
        y (Individual): Second vector.

    Returns:
        float: Chebyshev similarity between x and y.
    """
    if x.genome.shape != y.genome.shape:
        raise ValueError("Input vectors must have the same shape.")
    max_diff = np.max(np.abs(x.genome - y.genome))
    return 1 / (1 + max_diff)


def jensen_shannon_distance(x: Individual, y: Individual) -> float:
    """Compute the Jensen-Shannon distance between two probability distributions.

    Args:
        x (Individual): First probability distribution.
        y (Individual): Second probability distribution.

    Returns:
        float: Jensen-Shannon distance between x and y.
    """
    if x.genome.shape != y.genome.shape:
        raise ValueError("Input vectors must have the same shape.")
    if np.any(x.genome < 0) or np.any(y.genome < 0):
        raise ValueError("Input vectors must be non-negative.")
    if not np.isclose(np.sum(x.genome), 1) or not np.isclose(np.sum(y.genome), 1):
        raise ValueError("Input vectors must sum to 1.")
    m = 0.5 * (x.genome + y.genome)
    def kl_divergence(p, q):
        mask = p > 0
        return np.sum(p[mask] * np.log(p[mask] / q[mask]))
    return 0.5 * (kl_divergence(x.genome, m) + kl_divergence(y.genome, m)) ** 0.5
