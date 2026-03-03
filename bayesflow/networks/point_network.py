import keras
from typing import Sequence

from bayesflow.networks.scoring_rule_network import ScoringRuleNetwork
from bayesflow.scoring_rules import ScoringRule, MeanScore, QuantileScore


class PointNetwork(ScoringRuleNetwork):
    """
    (IN) Implements Bayesian estimation of point estimates like mean and quantiles using a
    shared feed-forward architecture.

    ``PointNetwork`` provides a subset of the functionality of :py:class:`ScoringRuleNetwork`
    with a simplified interface. It only supports a predefined set of scoring rules (currently
    mean and quantiles) and does not support custom scoring rules or parametric distribution scores.

    Examples
    --------
    The following two are equivalent:

    .. code-block:: pycon

        >>> inference_network = bf.networks.PointNetwork(["mean", "quantiles"], q=[0.1, 0.3, 0.5, 0.7, 0.9])

        >>> from bayesflow.scoring_rules import MeanScore, QuantileScore
        >>> inference_network = bf.networks.ScoringRuleNetwork(
        ...     mean=MeanScore(),
        ...     quantiles=QuantileScore([0.1, 0.3, 0.5, 0.7, 0.9]),
        ...     # mvn=MvNormalScore(),  # not supported by PointNetwork
        ... )

    ... but the latter supports passing any subclass of :py:class:`ScoringRule`, e.g. parametric distributions.
    """

    def __init__(
        self, points: str | Sequence[str], q: Sequence[float] | None = None, subnet: str | keras.Layer = "mlp", **kwargs
    ):
        scoring_rules = self._resolve_scoring_rules(points, q)
        super().__init__(scoring_rules=scoring_rules, subnet=subnet, **kwargs)

    def _resolve_scoring_rules(self, points: Sequence[str], q: Sequence[float]) -> dict[str, ScoringRule]:
        scoring_rules = {}

        if isinstance(points, str):
            points = [points]

        for p in points:
            match p:
                case "mean" as key:
                    scoring_rules[key] = MeanScore()
                case "quantiles" as key:
                    scoring_rules[key] = QuantileScore(q=q)
                case _ as key:
                    raise ValueError(f"{key} must be either `mean` or `quantiles`")

        return scoring_rules
