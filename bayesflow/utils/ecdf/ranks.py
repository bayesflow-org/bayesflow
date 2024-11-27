import numpy as np


def fractional_ranks(post_samples: np.ndarray, prior_samples: np.ndarray) -> np.ndarray:
    """Compute fractional ranks (using broadcasting)"""
    return np.mean(post_samples < prior_samples[:, np.newaxis, :], axis=1)


def _helper_distance_ranks(
    post_samples: np.ndarray,
    prior_samples: np.ndarray,
    stacked: bool,
    references: np.ndarray,
    distance: callable,
    p_norm: int,
) -> np.ndarray:
    """
    Helper function to compute ranks of true parameter wrt posterior samples
    based on distances (defined on the p_norm) between samples and a given references.
    """
    if distance is None:
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

    else:
        # compute distances using the given distance function
        if stacked:
            # compute distance over joint parameters
            dist_post = np.array([distance(post_samples[i], references[i]) for i in range(references.shape[0])])
            dist_prior = np.array([distance(prior_samples[i], references[i]) for i in range(references.shape[0])])
            ranks = np.mean((dist_post < dist_prior[:, np.newaxis]), axis=1)[:, np.newaxis]
        else:
            # compute distances per parameter
            dist_post = np.zeros_like(post_samples)
            dist_prior = np.zeros_like(prior_samples)
            for i in range(references.shape[0]):  # Iterate over samples
                for j in range(references.shape[1]):  # Iterate over parameters
                    dist_post[i, :, j] = distance(post_samples[i, :, j], references[i, j])
                    dist_prior[i, j] = distance(prior_samples[i, j], references[i, j])

            ranks = np.mean((dist_post < dist_prior[:, np.newaxis]), axis=1)
    return ranks


def distance_ranks(
    post_samples: np.ndarray,
    prior_samples: np.ndarray,
    stacked: bool,
    references: np.ndarray = None,
    distance: callable = None,
    p_norm: int = 2,
) -> np.ndarray:
    """
    Compute ranks of true parameter wrt posterior samples based on distances between samples and optional references.

    Parameters
    ----------
    post_samples : np.ndarray
        The posterior samples.
    prior_samples : np.ndarray
        The prior samples.
    references : np.ndarray, optional
        The references to compute the ranks.
    stacked : bool
        If True, compute ranks for all parameters jointly. Otherwise, compute marginal ranks.
    distance : callable, optional
        The distance function to compute the ranks. If None, the distance defined by the p_norm is used. Must be
        a function that takes two arrays (if stacked, it gets the full parameter vectors, if not only the single
        parameters) and returns an array with the distances. This could be based on the log-posterior, for example.
    p_norm : int, optional
        The norm to compute the distance if no distance is passed. Default is L2-norm.
    """
    # Reference is the origin
    if references is None:
        references = np.zeros((prior_samples.shape[0], prior_samples.shape[1]))
    else:
        # Validate reference
        if references.shape[0] != prior_samples.shape[0]:
            raise ValueError("The number of references must match the number of prior samples.")
        if references.shape[1] != prior_samples.shape[1]:
            raise ValueError("The dimension of references must match the dimension of the parameters.")

    ranks = _helper_distance_ranks(
        post_samples=post_samples,
        prior_samples=prior_samples,
        stacked=stacked,
        references=references,
        distance=distance,
        p_norm=p_norm,
    )
    return ranks
