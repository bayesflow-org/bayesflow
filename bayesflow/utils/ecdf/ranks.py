import numpy as np


def fractional_ranks(post_samples: np.ndarray, prior_samples: np.ndarray) -> np.ndarray:
    """Compute fractional ranks (using broadcasting)"""
    return np.mean(post_samples < prior_samples[:, np.newaxis, :], axis=1)


def _helper_distance_ranks(
    post_samples: np.ndarray, prior_samples: np.ndarray, references: np.ndarray, stacked: bool
) -> np.ndarray:
    """
    Helper function to compute ranks of true parameter wrt posterior samples
    based on distances between samples and a given references.

    """
    if stacked:
        # compute ranks for all parameters jointly
        samples_distances = np.sqrt(np.sum((references[:, np.newaxis, :] - post_samples) ** 2, axis=-1))
        theta_distances = np.sqrt(np.sum((references - prior_samples) ** 2, axis=-1))
        ranks = np.mean((samples_distances < theta_distances[:, np.newaxis]), axis=1)[:, np.newaxis]
    else:
        # compute marginal ranks for each parameter
        samples_distances = np.sqrt((references[:, np.newaxis, :] - post_samples) ** 2)
        theta_distances = np.sqrt((references - prior_samples) ** 2)
        ranks = np.mean((samples_distances < theta_distances[:, np.newaxis]), axis=1)
    return ranks


def distance_ranks(post_samples: np.ndarray, prior_samples: np.ndarray, stacked: bool) -> np.ndarray:
    """
    Compute ranks of true parameter wrt posterior samples based on distances between samples and the origin.
    """
    # Reference is the origin
    references = np.zeros((prior_samples.shape[0], prior_samples.shape[1]))
    ranks = _helper_distance_ranks(
        post_samples=post_samples, prior_samples=prior_samples, references=references, stacked=stacked
    )
    return ranks


def random_ranks(post_samples: np.ndarray, prior_samples: np.ndarray, stacked: bool) -> np.ndarray:
    """
    Compute ranks of true parameter wrt posterior samples based on distances between samples and random references.
    """
    # Create random references
    random_ref = np.random.uniform(low=-1, high=1, size=(prior_samples.shape[0], prior_samples.shape[-1]))
    half_size = random_ref.shape[0] // 2
    # We muss have a dependency on the true parameter otherwise potential biases will not be detected
    # the dependency of the first half of the references is on the one parameter, then on another
    references_1 = (
        np.tile(
            prior_samples[:, np.random.randint(prior_samples.shape[-1])],
            (prior_samples.shape[-1], 1),
        ).T[:half_size]
        + random_ref[:half_size]
    )

    # Create references for the second half
    references_2 = (
        np.tile(
            prior_samples[:, np.random.randint(prior_samples.shape[-1])],
            (prior_samples.shape[-1], 1),
        ).T[half_size:]
        + random_ref[half_size:]
    )
    references = np.concatenate((references_1, references_2), axis=0)
    ranks = _helper_distance_ranks(
        post_samples=post_samples, prior_samples=prior_samples, references=references, stacked=stacked
    )
    return ranks
