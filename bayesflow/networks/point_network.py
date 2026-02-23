import keras
from typing import Sequence
from bayesflow.networks.scoring_rule_inference_network import ScoringRuleNetwork
from bayesflow.scoring_rules import ScoringRule, MeanScoringRule, QuantileScoringRule


class PointNetwork(ScoringRuleNetwork):
    def __init__(
        self, points: Sequence[str], q: Sequence[float] | None = None, subnet: str | keras.Layer = "mlp", **kwargs
    ):
        scoring_rules = self._resolve_scoring_rules(points, q)
        super().__init__(scoring_rules, subnet, **kwargs)

    def _resolve_scoring_rules(self, points: Sequence[str], q) -> dict[str, ScoringRule]:
        scoring_rules = {}
        for p in points:
            match p:
                case "mean" as key:
                    scoring_rules[key] = MeanScoringRule()
                case "quantile" | "quantiles" as key:
                    scoring_rules[key] = QuantileScoringRule(q=q)
        return scoring_rules
