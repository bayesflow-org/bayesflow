from functools import singledispatch


@singledispatch
def find_scoring_rule(arg, *args, **kwargs):
    from bayesflow.networks.inference.scoring.scoring_rules import ScoringRule

    if isinstance(arg, ScoringRule):
        return arg
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
        case "leaky_exponential":
            from bayesflow.links import Leaky
            from bayesflow.scoring_rules import ExponentialScore

            kwargs.setdefault("links", {"logits": Leaky(power=2.0)})
            return ExponentialScore(*args, **kwargs)
        case "logistic":
            from bayesflow.scoring_rules import LogisticScore

            return LogisticScore(*args, **kwargs)
        case "power_logistic":
            from bayesflow.scoring_rules import LogisticScore

            kwargs.setdefault("alpha", 1.0)
            return LogisticScore(*args, **kwargs)
        case "mean":
            from bayesflow.scoring_rules import MeanScore

            return MeanScore(*args, **kwargs)
        case "median":
            from bayesflow.scoring_rules import MedianScore

            return MedianScore(*args, **kwargs)
        case "normed_difference":
            from bayesflow.scoring_rules import NormedDifferenceScore

            return NormedDifferenceScore(*args, **kwargs)
        case "quantile":
            from bayesflow.scoring_rules import QuantileScore

            return QuantileScore(*args, **kwargs)
        case "mv_normal" | "multivariate_normal":
            from bayesflow.scoring_rules import MvNormalScore

            return MvNormalScore(*args, **kwargs)
        case "mixture":
            from bayesflow.scoring_rules import MixtureScore

            return MixtureScore(*args, **kwargs)
        case other:
            raise ValueError(f"Unsupported scoring rule name: '{other}'.")


@find_scoring_rule.register(type)
def _(cls, *args, **kwargs):
    return cls(*args, **kwargs)
