import numpy as np


def fractional_ranks(post_samples: np.ndarray, prior_samples: np.ndarray) -> np.ndarray:
    """Compute fractional ranks (using broadcasting)"""
    return np.mean(post_samples < prior_samples[:, np.newaxis, :], axis=1)


def _helper_distance_ranks(
    post_samples: np.ndarray, prior_samples: np.ndarray, references: np.ndarray, stacked: bool, p_norm: int
) -> np.ndarray:
    """
    Helper function to compute ranks of true parameter wrt posterior samples
    based on distances (defined on the p_norm) between samples and a given references.
    """
    # compute distances to references
    dist_post = np.abs((references[:, np.newaxis, :] - post_samples))
    dist_prior = np.abs(references - prior_samples)

    if stacked:
        # compute ranks for all parameters jointly
        samples_distances = np.sum(dist_post**p_norm, axis=-1) ** (1 / p_norm)
        theta_distances = np.sum(dist_prior**p_norm, axis=-1) ** (1 / p_norm)

        ranks = np.mean((samples_distances < theta_distances[:, np.newaxis]), axis=1)[:, np.newaxis]
    else:
        # compute marginal ranks for each parameter
        ranks = np.mean((dist_post < dist_prior[:, np.newaxis]), axis=1)
    return ranks


def distance_ranks(post_samples: np.ndarray, prior_samples: np.ndarray, stacked: bool, p_norm: int = 2) -> np.ndarray:
    """
    Compute ranks of true parameter wrt posterior samples based on distances between samples and the origin.
    """
    # Reference is the origin
    references = np.zeros((prior_samples.shape[0], prior_samples.shape[1]))
    ranks = _helper_distance_ranks(
        post_samples=post_samples, prior_samples=prior_samples, references=references, stacked=stacked, p_norm=p_norm
    )
    return ranks


def reference_ranks(
    post_samples: np.ndarray, prior_samples: np.ndarray, references: np.ndarray, stacked: bool, p_norm: int = 2
) -> np.ndarray:
    """
    Compute ranks of true parameter wrt posterior samples based on distances between samples and references.
    """
    # Validate reference
    if references.shape[0] != prior_samples.shape[0]:
        raise ValueError("The number of references must match the number of prior samples.")
    if references.shape[1] != prior_samples.shape[1]:
        raise ValueError("The dimension of references must match the dimension of the parameters.")
    # Compute ranks
    ranks = _helper_distance_ranks(
        post_samples=post_samples, prior_samples=prior_samples, references=references, stacked=stacked, p_norm=p_norm
    )
    return ranks
