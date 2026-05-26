from functools import singledispatch

from bayesflow.networks.inference.scoring.scoring_rules import ScoringRule


@singledispatch
def find_scoring_rule(arg, *args, **kwargs) -> ScoringRule:
    raise TypeError(f"Cannot infer scoring rule from {arg!r}.")


@find_scoring_rule.register
def _(name: str, *args, **kwargs):
    match name.lower():
        case "cross_entropy" | "default":
            from bayesflow.scoring_rules import CrossEntropyScore

            return CrossEntropyScore(*args, **kwargs)
        case "brier":
            from bayesflow.scoring_rules import BrierScore

            return BrierScore(*args, **kwargs)
        case "polynomial":
            from bayesflow.scoring_rules import PolynomialScore

            return PolynomialScore(*args, **kwargs)
        case "exponential":
            from bayesflow.scoring_rules import ExponentialScore

            return ExponentialScore(*args, **kwargs)
        case "scaled_exponential":
            from bayesflow.scoring_rules import ScaledExponentialScore

            return ScaledExponentialScore(*args, **kwargs)
        case "logistic":
            from bayesflow.scoring_rules import LogisticScore

            return LogisticScore(*args, **kwargs)
        case "power_logistic":
            from bayesflow.scoring_rules import PowerLogisticScore

            return PowerLogisticScore(*args, **kwargs)
        case "leaky_exponential":
            from bayesflow.scoring_rules import LeakyExponentialScore

            return LeakyExponentialScore(*args, **kwargs)
        case other:
            raise ValueError(f"Unsupported scoring rule name: '{other}'.")


@find_scoring_rule.register
def _(cls: type, *args, **kwargs):
    return cls(*args, **kwargs)


@find_scoring_rule.register
def _(rule: ScoringRule, *args, **kwargs):
    return rule
