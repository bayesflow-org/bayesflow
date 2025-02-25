from collections.abc import Sequence

from bayesflow.types import Shape, Tensor

from bayesflow.links import OrderedQuantiles, PositiveSemiDefinite

from bayesflow.utils import logging, find_network, serialize_value_or_type, deserialize_value_or_type

import keras

import math


class ScoringRule:
    def __init__(
        self,
        subnets: dict[str, str | type] = None,
        subnets_kwargs: dict[str, dict] = None,
        links: dict[str, str | type] = None,
    ):
        self.subnets = subnets or {}
        self.subnets_kwargs = subnets_kwargs or {}
        self.links = links or {}

        self.config = {
            "subnets_kwargs": self.subnets_kwargs,
        }

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

    def get_head_shapes_from_target_shape(self, target_shape):
        raise NotImplementedError

    def set_head_shapes_from_target_shape(self, target_shape):
        self.head_shapes = self.get_head_shapes_from_target_shape(target_shape)

    def get_subnet(self, key: str):
        if key not in self.subnets.keys():
            return keras.layers.Identity()
        else:
            return find_network(self.subnets[key], **self.subnets_kwargs.get(key, {}))

    def get_link(self, key: str):
        if key not in self.links.keys():
            return keras.layers.Activation("linear")
        elif isinstance(self.links[key], str):
            return keras.layers.Activation(self.links[key])
        else:
            return self.links[key]

    def get_head(self, key: str):
        subnet = self.get_subnet(key)
        head_shape = self.head_shapes[key]
        dense = keras.layers.Dense(units=math.prod(head_shape))
        reshape = keras.layers.Reshape(target_shape=head_shape)
        link = self.get_link(key)
        return keras.Sequential([subnet, dense, reshape, link])

    def score(self, estimates: dict[str, Tensor], targets: Tensor, weights: Tensor) -> Tensor:
        raise NotImplementedError

    def aggregate(self, scores: Tensor, weights: Tensor = None):
        if weights is not None:
            weighted = scores * weights
        else:
            weighted = scores
        return keras.ops.mean(weighted)


class NormedDifferenceScore(ScoringRule):
    def __init__(
        self,
        k: int,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.k = k

        self.config = {
            "k": k,
        }

    def get_head_shapes_from_target_shape(self, target_shape):
        # keras.saving.load_model sometimes passes target_shape as a list.
        # This is why I force a conversion to tuple here.
        target_shape = tuple(target_shape)
        return dict(value=target_shape[1:])

    def score(self, estimates: dict[str, Tensor], targets: Tensor, weights: Tensor = None) -> Tensor:
        estimates = estimates["value"]
        pointwise_differance = estimates - targets
        scores = keras.ops.absolute(pointwise_differance) ** self.k
        score = self.aggregate(scores, weights)
        return score

    def get_config(self):
        base_config = super().get_config()
        return base_config | self.config


class MedianScore(NormedDifferenceScore):
    def __init__(self, **kwargs):
        super().__init__(k=1, **kwargs)
        self.config = {}


class MeanScore(NormedDifferenceScore):
    def __init__(self, **kwargs):
        super().__init__(k=2, **kwargs)
        self.config = {}


class QuantileScore(ScoringRule):
    def __init__(self, q: Sequence[float] = None, links=None, **kwargs):
        super().__init__(links=links, **kwargs)
        if q is None:
            q = [0.1, 0.5, 0.9]
            logging.info(f"QuantileScore was not provided with argument `q`. Using the default quantile levels: {q}.")

        self.q = q
        self._q = keras.ops.convert_to_tensor(q, dtype="float32")
        self.links = links or {"value": OrderedQuantiles(q=q)}

        self.config = {
            "q": q,
        }

    def get_config(self):
        base_config = super().get_config()
        return base_config | self.config

    def get_head_shapes_from_target_shape(self, target_shape):
        # keras.saving.load_model sometimes passes target_shape as a list.
        # This is why I force a conversion to tuple here.
        target_shape = tuple(target_shape)
        return dict(value=(len(self.q),) + target_shape[1:])

    def score(self, estimates: dict[str, Tensor], targets: Tensor, weights: Tensor = None) -> Tensor:
        estimates = estimates["value"]
        pointwise_differance = estimates - targets[:, None, :]

        scores = pointwise_differance * (keras.ops.cast(pointwise_differance > 0, float) - self._q[None, :, None])
        scores = keras.ops.mean(scores, axis=1)
        score = self.aggregate(scores, weights)
        return score


class ParametricDistributionRule(ScoringRule):
    """
    TODO
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def log_prob(self, x, **kwargs):
        raise NotImplementedError

    def sample(self, batch_shape, **kwargs):
        raise NotImplementedError

    def score(self, estimates: dict[str, Tensor], targets: Tensor, weights: Tensor = None) -> Tensor:
        scores = -self.log_prob(x=targets, **estimates)
        score = self.aggregate(scores, weights)
        # multipy to mitigate instability due to relatively high values of parametric score
        return score * 0.01


class MultivariateNormalScore(ParametricDistributionRule):
    def __init__(self, D: int = None, links=None, **kwargs):
        super().__init__(links=links, **kwargs)
        self.D = D
        self.links = links or {"covariance": PositiveSemiDefinite()}

        self.config = {
            "D": D,
        }

    def get_config(self):
        base_config = super().get_config()
        return base_config | self.config

    def get_head_shapes_from_target_shape(self, target_shape) -> dict[str, Shape]:
        self.D = target_shape[-1]
        return dict(
            mean=(self.D,),
            covariance=(self.D, self.D),
        )

    def log_prob(self, x: Tensor, mean: Tensor, covariance: Tensor) -> Tensor:
        diff = x[:, None, :] - mean
        inv_covariance = keras.ops.inv(covariance)
        log_det_covariance = keras.ops.slogdet(covariance)[1]  # Only take the log of the determinant part

        # Compute the quadratic term in the exponential of the multivariate Gaussian
        quadratic_term = keras.ops.einsum("...i,...ij,...j->...", diff, inv_covariance, diff)

        # Compute the log probability density
        log_prob = -0.5 * (self.D * keras.ops.log(2 * math.pi) + log_det_covariance + quadratic_term)

        return log_prob

    # WIP: incorrect draft
    def sample(self, batch_shape, mean, covariance):
        batch_size, D = mean.shape
        # Ensure covariance is (batch_size, D, D)
        assert covariance.shape == (batch_size, D, D)

        # Use Cholesky decomposition to generate samples
        chol = keras.ops.cholesky(covariance)
        normal_samples = keras.random.normal((*batch_shape, D))
        samples = mean[:, :, None] + keras.ops.einsum("...ijk,...ikl->...ijl", chol, normal_samples)

        return samples
