import pytest


@pytest.fixture(
    params=["median_score", "mean_score", "normed_diff_score", "quantile_score", "multivariate_normal_score"],
)
def scoring_rule(request):
    name, kwargs = request.param
    match name:
        case "median_score":
            from bayesflow.scores import MedianScore

            return MedianScore(**kwargs)
        case "mean_score":
            from bayesflow.scores import MeanScore

            return MeanScore(**kwargs)
        case "normed_diff_score":
            from bayesflow.scores import NormedDifferenceScore

            return NormedDifferenceScore(**kwargs)
        case "quantile_score":
            from bayesflow.scores import QuantileScore

            return QuantileScore(**kwargs)
        case "multivariate_normal_score":
            from bayesflow.scores import MultivariateNormalScore

            return MultivariateNormalScore(**kwargs)
        case _:
            raise ValueError(f"Invalid request parameter for scoring_rule: {name}")
