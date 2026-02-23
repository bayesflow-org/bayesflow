from typing import Sequence

import keras

from bayesflow.types import Shape, Tensor
from bayesflow.utils import logging, weighted_mean
from bayesflow.utils.serialization import serializable
from bayesflow.links import OrderedQuantiles

from .scoring_rule import ScoringRule


@serializable("bayesflow.scores")
class QuantileScore(ScoringRule):
    r""":math:`S(\hat \theta_i, \theta; \tau_i)
    = (\hat \theta_i - \theta)(\mathbf{1}_{\hat \theta - \theta > 0} - \tau_i)`

    Scores predicted quantiles :math:`\hat \theta_i` with the quantile score
    to match the quantile levels :math:`\hat \tau_i`.
    """

    def __init__(self, q: Sequence[float] = None, links=None, **kwargs):
        super().__init__(links=links, **kwargs)
        if q is None:
            q = [0.1, 0.5, 0.9]
            logging.info(f"QuantileScore was not provided with argument `q`. Using the default quantile levels: {q}.")

        # force a conversion to list for proper serialization
        q = list(q)
        self.q = q
        self._q = keras.ops.convert_to_tensor(q, dtype="float32")
        self.links = links or {"value": OrderedQuantiles(q=q)}

        self.config = {
            "q": q,
        }

    def get_config(self):
        base_config = super().get_config()
        return base_config | self.config

    def get_head_shapes_from_target_shape(self, target_shape: Shape) -> dict[str, tuple]:
        # keras.saving.load_model sometimes passes target_shape as a list, so we force a conversion
        target_shape = tuple(target_shape)
        return dict(value=(len(self.q),) + target_shape[1:])

    def score(self, estimates: dict[str, Tensor], targets: Tensor, weights: Tensor = None) -> Tensor:
        estimates = estimates["value"]
        pointwise_differance = estimates - targets[:, None, :]
        scores = pointwise_differance * (keras.ops.cast(pointwise_differance > 0, float) - self._q[None, :, None])
        scores = keras.ops.mean(scores, axis=1)
        score = weighted_mean(scores, weights)
        return score
