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
        case "scaled_exponential":
            from bayesflow.scoring_rules import ExponentialScore

            return ExponentialScore(*args, **kwargs)
        case "leaky_exponential":
            from bayesflow.scoring_rules import ExponentialScore

            kwargs.setdefault("leaky", 2.0)
            return ExponentialScore(*args, **kwargs)
        case "logistic":
            from bayesflow.scoring_rules import LogisticScore

            return LogisticScore(*args, **kwargs)
        case "power_logistic":
            from bayesflow.scoring_rules import LogisticScore

            kwargs.setdefault("alpha", 1.0)
            return LogisticScore(*args, **kwargs)
        case other:
            raise ValueError(f"Unsupported scoring rule name: '{other}'.")


@find_scoring_rule.register(type)
def _(cls, *args, **kwargs):
    return cls(*args, **kwargs)
