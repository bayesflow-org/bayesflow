import math

import keras

from bayesflow.types import Shape, Tensor
from bayesflow.utils import find_network
from bayesflow.utils.serialization import deserialize, serializable, serialize


@serializable("bayesflow.scores")
class ScoringRule:
    """Base class for scoring rules.

    Scoring rules evaluate the quality of statistical predictions based on the values that materialize
    when sampling from the true distribution. By minimizing an expected score, estimates with
    different properties can be obtained.

    To define a custom :py:class:`ScoringRule`, inherit from this class and overwrite the score method.
    For proper serialization, any new constructor arguments must be taken care of in a `get_config` method.

    Estimates are typically parameterized by projection heads consisting of a neural network component
    and a link to project into the correct output space.

    A :py:class:`ScoringRule` can score estimates consisting of multiple parts. See :py:class:`MultivariateNormalScore`
    for an example of a :py:class:`ParametricDistributionScore`. That score evaluates an estimated mean
    and covariance simultaneously.
    """

    NOT_TRANSFORMING_LIKE_VECTOR_WARNING: tuple[str] = tuple()
    """
    Names of prediction heads for which to warn if the adapter is called on their estimates in inverse direction.

    Prediction heads can output estimates in spaces other than the target distribution space.
    To such estimates the adapter cannot be straightforwardly applied in inverse direction,
    because the adapter is built to map vectors from the inference variable space. When subclassing
    :py:class:`ScoringRule`, add the names of such heads to the following list to warn users about difficulties
    with a type of estimate whenever the adapter is applied to them in inverse direction.
    """

    TRANSFORMATION_TYPE: dict[str, str] = {}
    """
    Defines nonstandard transformation behaviour for de-standardization.

    The standard transformation

    x_i = x_i' * sigma_i + mu_i

    is referred to as "location_scale". Keys not specified here will fallback to that default.
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
        self.config["subnets"] = {key: serialize(subnet) for key, subnet in self.subnets.items()}
        self.config["links"] = {key: serialize(link) for key, link in self.links.items()}

        return self.config

    @classmethod
    def from_config(cls, config):
        config = config.copy()
        config["subnets"] = {key: deserialize(subnet_dict) for key, subnet_dict in config["subnets"].items()}
        config["links"] = {key: deserialize(link_dict) for key, link_dict in config["links"].items()}

        return cls(**config)

    def get_head_shapes_from_target_shape(self, target_shape: Shape) -> dict[str, Shape]:
        """Request a dictionary of names and output shapes of required heads from the score."""
        raise NotImplementedError

    def get_subnet(self, key: str) -> keras.Layer:
        """For a specified key, request a subnet to be used for projecting the shared condition embedding
        before further projection and reshaping to the heads output shape.

        If no subnet was specified for the key (e.g. upon initialization),
        return just an instance of keras.layers.Identity.

        Parameters
        ----------
        key : str
            Name of head for which to request a subnet.

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

        If no link was specified for the key (e.g. upon initialization), return a linear activation.

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

    def get_head(self, key: str, output_shape: Shape) -> keras.Sequential:
        """For a specified head key and output shape, request corresponding head network.

        A head network has the following components that are called sequentially:

        1. subnet: A keras.Layer.
        2. dense: A trainable linear projection with as many units as are required by the next component.
        3. reshape: Changes shape of output of projection to match requirements of next component.
        4. link: Transforms unconstrained values into a constrained space for the final estimator.
           See :py:mod:`~bayesflow.links` for examples.

        This method initializes the components in reverse order to meet all requirements and returns them.

        Parameters
        ----------
        key : str
            Name of head for which to request a link.
        output_shape: Shape
            The necessary shape of estimated values for the given key as returned by
            :py:func:`get_head_shapes_from_target_shape()`.

        Returns
        -------
        head : keras.Sequential
            Head network consisting of a learnable projection, a reshape and a link operation
            to parameterize estimates.
        """
        # initialize head components back to front
        link = self.get_link(key)

        # link input shape can differ from output shape
        if hasattr(link, "compute_input_shape"):
            link_input_shape = link.compute_input_shape(output_shape)
        else:
            link_input_shape = output_shape

        reshape = keras.layers.Reshape(target_shape=link_input_shape)
        dense = keras.layers.Dense(units=math.prod(link_input_shape))
        subnet = self.get_subnet(key)

        return keras.Sequential([subnet, dense, reshape, link])

    def score(self, estimates: dict[str, Tensor], targets: Tensor, weights: Tensor) -> Tensor:
        r"""Scores a batch of probabilistic estimates of distributions based on samples
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
        The following shows how to score estimates with a :py:class:`MeanScore`. All :py:class:`ScoringRule`\ s
        follow this pattern, only differing in the structure of the estimates dictionary.

        >>> import keras
        >>> from bayesflow.scores import MeanScore
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
