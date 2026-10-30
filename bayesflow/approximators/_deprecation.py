from warnings import warn


def warn_approximator_deprecation(old_name: str, new_name: str) -> None:
    warn(
        f"`bayesflow.approximators.{old_name}` is deprecated. Use `bayesflow.estimators.{new_name}` instead.",
        DeprecationWarning,
        stacklevel=3,
    )
