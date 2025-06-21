import pytest
from tests.utils import assert_allclose
import keras


def test_valid_summaries(approximator_with_summaries, mean_std_summary_network, monkeypatch):
    monkeypatch.setattr(approximator_with_summaries, "summary_network", mean_std_summary_network)
    summaries = approximator_with_summaries.summarize({"summary_variables": keras.ops.ones((2, 3))})
    assert_allclose(summaries, keras.ops.stack([keras.ops.ones((2,)), keras.ops.zeros((2,))], axis=-1))


def test_no_summary_network(approximator_with_summaries, monkeypatch):
    monkeypatch.setattr(approximator_with_summaries, "summary_network", None)

    with pytest.raises(ValueError):
        approximator_with_summaries.summarize({"summary_variables": keras.ops.ones((2, 3))})


def test_no_summary_variables(approximator_with_summaries, mean_std_summary_network, monkeypatch):
    monkeypatch.setattr(approximator_with_summaries, "summary_network", mean_std_summary_network)

    with pytest.raises(ValueError):
        approximator_with_summaries.summarize({})
