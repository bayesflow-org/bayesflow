from bayesflow.types import Tensor

from .scoring_rule import ScoringRule


class CategoricalScoringRule(ScoringRule):
    """Base class for scoring rules over categorical (one-hot encoded) targets.

    All rules output length-``M`` logits. PMP rules estimate the posterior over
    models directly; Bayes factor rules (``is_pmp_rule = False``) additionally
    report log Bayes factors relative to a reference model.

    This is the expected base class for scoring rules passed to
    :class:`~bayesflow.approximators.ModelComparisonApproximator`.
    """

    # True for PMP rules, False for Bayes factor rules.
    is_pmp_rule: bool = True

    def to_log_odds(self, rule_output: dict[str, Tensor]) -> Tensor:
        """Map head output to length-``M`` log posterior odds."""
        return rule_output["logits"]
