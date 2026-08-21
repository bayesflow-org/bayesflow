import keras
import pytest

from bayesflow.utils import find_inference_network, find_distribution, find_network, find_summary_network
from bayesflow.networks.inference.diffusion import find_noise_schedule

# --- Tests for find__network.py ---


class DummyNetwork:
    def __init__(self, *a, **kw):
        self.args = a
        self.kwargs = kw


@pytest.mark.parametrize(
    "name,expected_class_path",
    [
        ("mlp", "bayesflow.networks.MLP"),
        ("time_mlp", "bayesflow.networks.TimeMLP"),
        ("diffusion_transformer", "bayesflow.networks.DiffusionTransformer"),
    ],
)
def test_find_network_by_name(monkeypatch, name, expected_class_path):
    # patch the expected class in bayesflow.networks
    components = expected_class_path.split(".")
    module_path = ".".join(components[:-1])
    class_name = components[-1]

    dummy_cls = DummyNetwork
    monkeypatch.setattr(f"{module_path}.{class_name}", dummy_cls)

    net = find_network(name, 1, key="val")
    assert isinstance(net, DummyNetwork)
    assert net.args == (1,)
    assert net.kwargs == {"key": "val"}


def test_find_network_by_type():
    # patch the expected class in bayesflow.networks
    net = find_network(DummyNetwork, 1, key="val")
    assert isinstance(net, DummyNetwork)
    assert net.args == (1,)
    assert net.kwargs == {"key": "val"}


def test_find_network_by_keras_layer():
    layer = keras.layers.Dense(10)
    result = find_network(layer)
    assert result is layer


def test_find_network_by_keras_model():
    model = keras.models.Sequential()
    result = find_network(model)
    assert result is model


def test_find_network_unknown_name():
    with pytest.raises(ValueError):
        find_network("unknown_network_name")


def test_find_network_invalid_type():
    with pytest.raises(TypeError):
        find_network(12345)


# --- Tests for find_inference_network.py ---


class DummyInferenceNetwork:
    def __init__(self, *a, **kw):
        self.args = a
        self.kwargs = kw


@pytest.mark.parametrize(
    "name,expected_class_path",
    [
        ("coupling_flow", "bayesflow.networks.CouplingFlow"),
        ("flow_matching", "bayesflow.networks.FlowMatching"),
        ("consistency_model", "bayesflow.networks.ConsistencyModel"),
    ],
)
def test_find_inference_network_by_name(monkeypatch, name, expected_class_path):
    # patch the expected class in bayesflow.networks
    components = expected_class_path.split(".")
    module_path = ".".join(components[:-1])
    class_name = components[-1]

    dummy_cls = DummyInferenceNetwork
    monkeypatch.setattr(f"{module_path}.{class_name}", dummy_cls)

    net = find_inference_network(name, 1, key="val")
    assert isinstance(net, DummyInferenceNetwork)
    assert net.args == (1,)
    assert net.kwargs == {"key": "val"}


def test_find_inference_network_by_type():
    # patch the expected class in bayesflow.networks
    net = find_inference_network(DummyInferenceNetwork, 1, key="val")
    assert isinstance(net, DummyInferenceNetwork)
    assert net.args == (1,)
    assert net.kwargs == {"key": "val"}


def test_find_inference_network_by_keras_layer():
    layer = keras.layers.Dense(10)
    result = find_inference_network(layer)
    assert result is layer


def test_find_inference_network_by_keras_model():
    model = keras.models.Sequential()
    result = find_inference_network(model)
    assert result is model


def test_find_inference_network_unknown_name():
    with pytest.raises(ValueError):
        find_inference_network("unknown_network_name")


def test_find_inference_network_invalid_type():
    with pytest.raises(TypeError):
        find_inference_network(12345)


# --- Tests for find_distribution.py ---


class DummyDistribution:
    def __init__(self, *a, **kw):
        self.args = a
        self.kwargs = kw


@pytest.mark.parametrize(
    "name, expected_class_path",
    [
        ("normal", "bayesflow.distributions.DiagonalNormal"),
        ("student", "bayesflow.distributions.DiagonalStudentT"),
        ("student-t", "bayesflow.distributions.DiagonalStudentT"),
        ("student_t", "bayesflow.distributions.DiagonalStudentT"),
    ],
)
def test_find_distribution_by_name(monkeypatch, name, expected_class_path):
    components = expected_class_path.split(".")
    module_path = ".".join(components[:-1])
    class_name = components[-1]

    dummy_cls = DummyDistribution
    monkeypatch.setattr(f"{module_path}.{class_name}", dummy_cls)

    dist = find_distribution(name, 10, a=5)
    assert isinstance(dist, DummyDistribution)
    assert dist.args == (10,)
    assert dist.kwargs == {"a": 5}


def test_find_distribution_none_returns_none():
    assert find_distribution(None) is None


def test_find_distribution_with_keras_layer():
    layer = keras.layers.Dense(3)
    result = find_distribution(layer)
    assert result is layer


def test_find_distribution_mixture_raises():
    with pytest.raises(ValueError):
        find_distribution("mixture")


def test_find_distribution_invalid_name():
    with pytest.raises(ValueError):
        find_distribution("invalid_name")


def test_find_distribution_invalid_type():
    with pytest.raises(TypeError):
        find_distribution(3.14)


# --- Tests for find_summary_network.py ---


class DummySummaryNetwork:
    def __init__(self, *a, **kw):
        self.args = a
        self.kwargs = kw


@pytest.mark.parametrize(
    "name,expected_class_path",
    [
        ("deep_set", "bayesflow.networks.DeepSet"),
        ("set_transformer", "bayesflow.networks.SetTransformer"),
        ("fusion_transformer", "bayesflow.networks.FusionTransformer"),
        ("time_series_transformer", "bayesflow.networks.TimeSeriesTransformer"),
        ("time_series_network", "bayesflow.networks.TimeSeriesNetwork"),
    ],
)
def test_find_summary_network_by_name(monkeypatch, name, expected_class_path):
    components = expected_class_path.split(".")
    module_path = ".".join(components[:-1])
    class_name = components[-1]

    dummy_cls = DummySummaryNetwork
    monkeypatch.setattr(f"{module_path}.{class_name}", dummy_cls)

    net = find_summary_network(name, 22, flag=True)
    assert isinstance(net, DummySummaryNetwork)
    assert net.args == (22,)
    assert net.kwargs == {"flag": True}


def test_find_summary_network_by_type():
    # patch the expected class in bayesflow.networks
    net = find_summary_network(DummySummaryNetwork, 1, key="val")
    assert isinstance(net, DummySummaryNetwork)
    assert net.args == (1,)
    assert net.kwargs == {"key": "val"}


def test_find_summary_network_by_keras_layer():
    layer = keras.layers.Dense(1)
    out = find_summary_network(layer)
    assert out is layer


def test_find_summary_network_by_keras_model():
    model = keras.models.Sequential()
    out = find_summary_network(model)
    assert out is model


def test_find_summary_network_unknown_name():
    with pytest.raises(ValueError):
        find_summary_network("unknown_summary_net")


def test_find_summary_network_invalid_type():
    with pytest.raises(TypeError):
        find_summary_network(0.1234)


def test_find_noise_schedule_by_name():
    from bayesflow.networks.inference.diffusion.schedules import CosineNoiseSchedule, EDMNoiseSchedule

    schedule = find_noise_schedule("cosine")
    assert isinstance(schedule, CosineNoiseSchedule)

    schedule = find_noise_schedule("edm")
    assert isinstance(schedule, EDMNoiseSchedule)


def test_find_noise_schedule_unknown_name():
    with pytest.raises(ValueError):
        find_noise_schedule("unknown_noise_schedule")


def test_pass_noise_schedule():
    from bayesflow.networks.inference.diffusion.schedules.noise_schedule import NoiseSchedule

    class CustomNoiseSchedule(NoiseSchedule):
        def __init__(self):
            pass

        def get_log_snr(self, t, training):
            pass

        def get_t_from_log_snr(self, log_snr_t, training):
            pass

        def derivative_log_snr(self, log_snr_t, training):
            pass

    schedule = CustomNoiseSchedule()
    assert schedule is find_noise_schedule(schedule)


def test_pass_noise_schedule_type():
    from bayesflow.networks.inference.diffusion.schedules import EDMNoiseSchedule

    schedule = find_noise_schedule(EDMNoiseSchedule, sigma_data=10.0)
    assert isinstance(schedule, EDMNoiseSchedule)
    assert schedule.sigma_data == 10.0


def test_find_noise_schedule_invalid_class():
    with pytest.raises(TypeError):
        find_noise_schedule(int)


def test_find_noise_schedule_invalid_object():
    with pytest.raises(TypeError):
        find_noise_schedule(1.0)


# --- Tests for find_scoring_rule.py ---


@pytest.mark.parametrize(
    "name,expected_class",
    [
        ("cross_entropy", "bayesflow.scoring_rules.CrossEntropyScore"),
        ("default", "bayesflow.scoring_rules.CrossEntropyScore"),
        ("brier", "bayesflow.scoring_rules.BrierScore"),
        ("polynomial", "bayesflow.scoring_rules.PolynomialScore"),
        ("exponential", "bayesflow.scoring_rules.ExponentialScore"),
        ("leaky_exponential", "bayesflow.scoring_rules.ExponentialScore"),
        ("logistic", "bayesflow.scoring_rules.LogisticScore"),
        ("power_logistic", "bayesflow.scoring_rules.LogisticScore"),
    ],
)
def test_find_scoring_rule_by_name(name, expected_class):
    from bayesflow.utils import find_scoring_rule
    import importlib

    module_path, class_name = expected_class.rsplit(".", 1)
    expected_cls = getattr(importlib.import_module(module_path), class_name)

    rule = find_scoring_rule(name)
    assert isinstance(rule, expected_cls)


def test_find_scoring_rule_leaky_default_sets_link():
    from bayesflow.links import Leaky
    from bayesflow.scoring_rules import ExponentialScore
    from bayesflow.utils import find_scoring_rule

    rule = find_scoring_rule("leaky_exponential")
    assert isinstance(rule, ExponentialScore)
    link = rule.get_link("logits")
    assert isinstance(link, Leaky)
    assert link.power == 2.0


def test_find_scoring_rule_power_logistic_default_sets_alpha():
    from bayesflow.scoring_rules import LogisticScore
    from bayesflow.utils import find_scoring_rule

    rule = find_scoring_rule("power_logistic")
    assert isinstance(rule, LogisticScore)
    assert rule.alpha == 1.0


def test_find_scoring_rule_by_type():
    from bayesflow.utils import find_scoring_rule
    from bayesflow.scoring_rules import CrossEntropyScore

    rule = find_scoring_rule(CrossEntropyScore)
    assert isinstance(rule, CrossEntropyScore)


def test_find_scoring_rule_by_instance():
    from bayesflow.utils import find_scoring_rule
    from bayesflow.scoring_rules import BrierScore

    existing = BrierScore()
    result = find_scoring_rule(existing)
    assert result is existing


def test_find_scoring_rule_unknown_name():
    from bayesflow.utils import find_scoring_rule

    with pytest.raises(ValueError, match="Unsupported scoring rule name"):
        find_scoring_rule("unknown_rule")


def test_find_scoring_rule_invalid_type():
    from bayesflow.utils import find_scoring_rule

    with pytest.raises(TypeError):
        find_scoring_rule(42)


@pytest.mark.parametrize(
    "name,expected_class",
    [
        ("mean", "bayesflow.scoring_rules.MeanScore"),
        ("median", "bayesflow.scoring_rules.MedianScore"),
        ("quantile", "bayesflow.scoring_rules.QuantileScore"),
        ("mv_normal", "bayesflow.scoring_rules.MvNormalScore"),
        ("multivariate_normal", "bayesflow.scoring_rules.MvNormalScore"),
    ],
)
def test_find_scoring_rule_continuous_by_name(name, expected_class):
    import importlib

    from bayesflow.utils import find_scoring_rule

    module_path, class_name = expected_class.rsplit(".", 1)
    expected_cls = getattr(importlib.import_module(module_path), class_name)
    rule = find_scoring_rule(name)
    assert isinstance(rule, expected_cls)


def test_find_scoring_rule_normed_difference():
    from bayesflow.scoring_rules import NormedDifferenceScore
    from bayesflow.utils import find_scoring_rule

    rule = find_scoring_rule("normed_difference", k=2)
    assert isinstance(rule, NormedDifferenceScore)


def test_find_scoring_rule_mixture():
    from bayesflow.scoring_rules import MixtureScore, MvNormalScore
    from bayesflow.utils import find_scoring_rule

    rule = find_scoring_rule("mixture", mvn1=MvNormalScore(), mvn2=MvNormalScore())
    assert isinstance(rule, MixtureScore)


def test_find_scoring_rule_case_insensitive():
    from bayesflow.scoring_rules import CrossEntropyScore
    from bayesflow.utils import find_scoring_rule

    rule = find_scoring_rule("CROSS_ENTROPY")
    assert isinstance(rule, CrossEntropyScore)
