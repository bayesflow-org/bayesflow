import math

import keras

from bayesflow.types import Shape, Tensor
from bayesflow.utils import find_network, serialize_value_or_type, deserialize_value_or_type


class ScoringRule:
    """Base class for scoring rules.

    Scoring rules evaluate the quality of statistical predictions based on the values that materialize
    when sampling from the true distribution. By minimizing an expected score, estimates with
    different properties can be obtained.

    To define a custom ``ScoringRule``, inherit from this class and overwrite the score method.
    For proper serialization, any new constructor arguments must be taken care of in a `get_config` method.
    """

    def __init__(
        self,
        subnets: dict[str, str | type] = None,
        subnets_kwargs: dict[str, dict] = None,
        links: dict[str, str | type] = None,
    ):
        self.subnets = subnets or {}
        self.subnets_kwargs = subnets_kwargs or {}
        self.links = links or {}

        self.config = {"subnets_kwargs": self.subnets_kwargs}

    def get_config(self):
        self.config["subnets"] = {
            key: serialize_value_or_type({}, "subnet", subnet) for key, subnet in self.subnets.items()
        }
        self.config["links"] = {key: serialize_value_or_type({}, "link", link) for key, link in self.links.items()}

        return self.config

    @classmethod
    def from_config(cls, config):
        config = config.copy()
        config["subnets"] = {
            key: deserialize_value_or_type(subnet_dict, "subnet")["subnet"]
            for key, subnet_dict in config["subnets"].items()
        }
        config["links"] = {
            key: deserialize_value_or_type(link_dict, "link")["link"] for key, link_dict in config["links"].items()
        }

        return cls(**config)

    def get_head_shapes_from_target_shape(self, target_shape: Shape) -> dict[str, Shape]:
        """Request a dictionary of names and output shapes of required heads from the score."""
        raise NotImplementedError

    def get_subnet(self, key: str) -> keras.Layer:
        """For a specified key, request a subnet to be used for projecting the shared condition embedding
        before reshaping to the heads output shape.

        Parameters
        ----------
        key : str
            Name of head for which to request a link.

        Returns
        -------
        link : keras.Layer
            Subnet projecting the shared condition embedding.
        """
        if key not in self.subnets.keys():
            return keras.layers.Identity()
        else:
            return find_network(self.subnets[key], **self.subnets_kwargs.get(key, {}))

    def get_link(self, key: str) -> keras.Layer:
        """For a specified key, request a link from network output to estimation target.

        Parameters
        ----------
        key : str
            Name of head for which to request a link.

        Returns
        -------
        link : keras.Layer
            Activation function linking network output to estimation target.
        """
        if key not in self.links.keys():
            return keras.layers.Activation("linear")
        elif isinstance(self.links[key], str):
            return keras.layers.Activation(self.links[key])
        else:
            return self.links[key]

    def get_head(self, key: str, shape: Shape) -> keras.Sequential:
        """For a specified head key and shape, request corresponding head network.

        Parameters
        ----------
        key : str
            Name of head for which to request a link.
        shape: Shape
            The necessary shape for the point estimators.

        Returns
        -------
        head : keras.Sequential
            Head network consisting of a learnable projection, a reshape and a link operation
            to parameterize estimates.
        """
        subnet = self.get_subnet(key)
        dense = keras.layers.Dense(units=math.prod(shape))
        reshape = keras.layers.Reshape(target_shape=shape)
        link = self.get_link(key)
        return keras.Sequential([subnet, dense, reshape, link])

    def score(self, estimates: dict[str, Tensor], targets: Tensor, weights: Tensor) -> Tensor:
        """Scores a batch of probabilistic estimates of distributions based on samples
        of the corresponding distributions.

        Parameters
        ----------
        estimates : dict[str, Tensor]
            Dictionary of estimates.
        targets : Tensor
            Array of samples from the true distribution to evaluate the estimates.
        weights : Tensor
            Array of weights for aggregating the scores.

        Returns
        -------
        numeric_score : Tensor
            Negatively oriented score evaluating the estimates, aggregated for the whole batch.

        Examples
        --------
        The following shows how to score estimates with a ``MeanScore``. All ``ScoringRule`` s follow this pattern,
        only differing in the structure of the estimates dictionary.

        >>> import keras
        ... from bayesflow.scores import MeanScore
        >>>
        >>> # batch of samples from a normal distribution
        >>> samples = keras.random.normal(shape=(100,))
        >>>
        >>> # batch of uninformed (random) estimates
        >>> bad_estimates = {"value": keras.random.uniform((100,))}
        >>>
        >>> # batch of estimates that are closer to the true mean
        >>> better_estimates = {"value": keras.random.normal(stddev=0.1, shape=(100,))}
        >>>
        >>> # calculate the score
        >>> scoring_rule = MeanScore()
        >>> scoring_rule.score(bad_estimates, samples)
        <tf.Tensor: shape=(), dtype=float32, numpy=1.2243813276290894>
        >>> scoring_rule.score(better_estimates, samples)
        <tf.Tensor: shape=(), dtype=float32, numpy=1.013983130455017>
        """
        raise NotImplementedError

    def aggregate(self, scores: Tensor, weights: Tensor = None) -> Tensor:
        """
        Computes the mean of scores, optionally applying weights.

        This function computes the mean value of the given scores. When weights are provided,
        it first multiplies the scores by the weights and then computes the mean of the result.
        If no weights are provided, it computes the mean of the scores.

        Parameters
        ----------
        scores : Tensor
            A tensor containing the scores to be aggregated.
        weights : Tensor, optional (default - None)
            A tensor of weights corresponding to each score. Must be the same shape as `scores`.
            If not provided, the function returns the mean of `scores`.

        Returns
        -------
        Tensor
            The aggregated score computed as a weighted mean if `weights` is provided,
            or as the simple mean of `scores` otherwise.
        """

        if weights is not None:
            return keras.ops.mean(scores * weights)
        return keras.ops.mean(scores)
