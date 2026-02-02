import keras
from typing import Sequence
from bayesflow.networks.scoring_rule_inference_network import ScoringRuleInferenceNetwork
from bayesflow.scores import ScoringRule, MeanScore, QuantileScore


class PointNetwork(ScoringRuleInferenceNetwork):
    def __init__(
        self, points: Sequence[str], q: Sequence[float] | None = None, subnet: str | keras.Layer = "mlp", **kwargs
    ):
        scores = self._resolve_scores(points, q)
        super().__init__(scores, subnet, **kwargs)

    def _resolve_scores(self, points: Sequence[str], q) -> dict[str, ScoringRule]:
        scores = {}
        for p in points:
            match p:
                case "mean" as key:
                    scores[key] = MeanScore()
                case "quantile" | "quantiles" as key:
                    scores[key] = QuantileScore(q=q)
        return scores
