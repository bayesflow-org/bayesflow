from bayesflow.types import Tensor

from .scoring_rule import ScoringRule


class CategoricalScoringRule(ScoringRule):
    """Base class for scoring rules over categorical (one-hot encoded) targets.

    Subclasses estimate either posterior model probabilities (PMP rules, output
    logits) or log Bayes factors (BF rules, output log Bayes factors relative to
    a reference model).

    This is the expected base class for scoring rules passed to
    :class:`~bayesflow.approximators.ModelComparisonApproximator`.
    """

    @property
    def is_pmp_rule(self) -> bool:
        """True for PMP rules (output logits), False for Bayes factor rules (output log Bayes factors)."""
        return "logits" in self.get_head_shapes_from_target_shape((1, 2))

    def to_bayes_factors(self, f: Tensor) -> Tensor:
        """Convert raw network outputs to log Bayes factors.

        The default implementation is the identity, correct for BF rules whose
        minimizer is directly the log Bayes factor (e.g. :class:`ExponentialScore`
        and :class:`LogisticScore` at their default settings). Subclasses whose
        minimizer lives in a transformed space must override this method.
        """
        return f
