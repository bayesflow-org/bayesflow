from bayesflow.types import Tensor

from .scoring_rule import ScoringRule


class CategoricalScoringRule(ScoringRule):
    """Base class for scoring rules over categorical (one-hot encoded) targets.

    This is the expected base class for scoring rules passed to
    :class:`~bayesflow.approximators.ModelComparisonApproximator`.
    """

    def to_log_odds(self, rule_output: dict[str, Tensor]) -> Tensor:
        """Map head output to M log posterior odds."""
        return rule_output["logits"]
