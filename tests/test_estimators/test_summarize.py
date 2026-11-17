import pytest
from tests.utils import assert_allclose
import keras


def test_valid_summaries(estimator_with_summaries, mean_std_summary_network, monkeypatch):
    monkeypatch.setattr(estimator_with_summaries, "summary_network", mean_std_summary_network)
    estimator_with_summaries.standardizer.build(
        {"inference_variables": (2, 2), "inference_conditions": (2, 3), "summary_variables": (2, 3)}
    )

    summaries = estimator_with_summaries.summarize({"summary_variables": keras.ops.ones((2, 3))})
    assert_allclose(summaries, keras.ops.stack([keras.ops.ones((2,)), keras.ops.zeros((2,))], axis=-1))

    summaries = estimator_with_summaries.summarize({"summary_variables": keras.ops.ones((2, 3))}, batch_size=1)
    assert_allclose(summaries, keras.ops.stack([keras.ops.ones((2,)), keras.ops.zeros((2,))], axis=-1))


def test_no_summary_network(estimator_with_summaries, monkeypatch):
    monkeypatch.setattr(estimator_with_summaries, "summary_network", None)

    with pytest.raises(ValueError):
        estimator_with_summaries.summarize({"summary_variables": keras.ops.ones((2, 3))})


def test_no_summary_variables(estimator_with_summaries, mean_std_summary_network, monkeypatch):
    monkeypatch.setattr(estimator_with_summaries, "summary_network", mean_std_summary_network)

    with pytest.raises(ValueError):
        estimator_with_summaries.summarize({})
