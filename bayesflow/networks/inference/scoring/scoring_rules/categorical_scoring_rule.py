from collections.abc import Mapping

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


def check_categorical_rules_not_mixed(scoring_rules) -> None:
    """Raise if PMP and Bayes factor categorical scoring rules are mixed together.

    PMP rules (which output posterior model probabilities) and Bayes factor rules (which
    output log Bayes factors) estimate different quantities and cannot be co-trained or
    merged in the same model. Only :class:`CategoricalScoringRule` instances are inspected;
    other scoring rules (e.g. point-estimation rules) are ignored, and the check is a no-op
    unless both families are present.

    Parameters
    ----------
    scoring_rules : Mapping[str, ScoringRule] or Iterable[ScoringRule]
        The scoring rules to validate.

    Raises
    ------
    ValueError
        If at least one PMP rule and one Bayes factor rule are present.
    """
    items = scoring_rules.items() if isinstance(scoring_rules, Mapping) else enumerate(scoring_rules)
    pmp, bf = [], []
    for key, rule in items:
        if isinstance(rule, CategoricalScoringRule):
            (pmp if rule.is_pmp_rule else bf).append(key)
    if pmp and bf:
        raise ValueError(
            "Cannot mix PMP and Bayes factor scoring rules in the same model. "
            f"PMP rules: {pmp}; Bayes factor rules: {bf}. "
            "Use one family of scoring rules (all PMP or all Bayes factor)."
        )
