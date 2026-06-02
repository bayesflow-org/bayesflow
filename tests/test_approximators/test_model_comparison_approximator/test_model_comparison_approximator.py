import keras
import io
import pytest
from contextlib import redirect_stdout

from tests.utils import assert_models_equal


def test_build(approximator, train_dataset):
    assert approximator.built is False

    data_shapes = keras.tree.map_structure(keras.ops.shape, train_dataset[0])
    approximator.build(data_shapes)

    assert approximator.built is True
    assert approximator.inference_network.built is True
    if approximator.summary_network is not None:
        assert approximator.summary_network.built is True


def test_build_adapter():
    from bayesflow.approximators import ModelComparisonApproximator

    _ = ModelComparisonApproximator.build_adapter(
        inference_conditions=["foo", "bar"],
        summary_variables=["observables"],
        inference_variables=["indices"],
    )


def test_build_dataset(approximator, simulator, adapter):
    from bayesflow.datasets import OnlineDataset

    dataset = approximator.build_dataset(
        simulator=simulator,
        adapter=adapter,
        memory_budget="20 KiB",
        num_batches=2,
    )
    assert isinstance(dataset, OnlineDataset)


def test_fit(approximator, train_dataset, validation_dataset):
    approximator.compile(optimizer="AdamW")
    num_epochs = 1

    # Capture ostream and train model
    with io.StringIO() as stream:
        with redirect_stdout(stream):
            approximator.fit(dataset=train_dataset, validation_data=validation_dataset, epochs=num_epochs)

        output = stream.getvalue()
    # check that the loss is shown
    assert "loss" in output


def test_save_and_load(tmp_path, approximator, train_dataset, validation_dataset):
    # to save, the model must be built
    data_shapes = keras.tree.map_structure(keras.ops.shape, train_dataset[0])
    approximator.build(data_shapes)
    approximator.compute_metrics(**train_dataset[0])

    keras.saving.save_model(approximator, tmp_path / "model.keras")
    loaded = keras.saving.load_model(tmp_path / "model.keras")

    assert_models_equal(approximator, loaded)


def test_estimate(approximator, train_dataset, simulator):
    data_shapes = keras.tree.map_structure(keras.ops.shape, train_dataset[0])
    approximator.build(data_shapes)
    approximator.compute_metrics(**train_dataset[0])

    num_conditions = 2
    num_models = len(simulator.simulators)
    conditions = simulator.sample(num_conditions)
    output = approximator.estimate(conditions=conditions)

    assert isinstance(output, dict)
    assert "model_probs" in output
    assert output["model_probs"].shape == (num_conditions, num_models)

    if approximator.scoring_rule.is_pmp_rule:
        assert "logits" in output
        assert output["logits"].shape == (num_conditions, num_models)
    else:
        assert "log_bayes_factors" in output
        assert output["log_bayes_factors"].shape == (num_conditions, num_models - 1)

    if approximator.summary_network is not None:
        assert "_summaries" in output
        assert output["_summaries"].ndim == 2
        assert output["_summaries"].shape[0] == num_conditions
    else:
        assert "_summaries" not in output


def test_rejects_non_categorical_scoring_rule():
    import keras
    from bayesflow.approximators import ModelComparisonApproximator
    from bayesflow.scoring_rules import MeanScore

    with pytest.raises(TypeError, match="CategoricalScoringRule"):
        ModelComparisonApproximator(
            num_models=2,
            classifier_network=keras.layers.Dense(4),
            scoring_rules=MeanScore(),
        )


def test_is_pmp_rule_property(approximator):
    from bayesflow.scoring_rules import CrossEntropyScore, ExponentialScore

    if isinstance(approximator.scoring_rule, CrossEntropyScore):
        assert approximator.scoring_rule.is_pmp_rule is True
    elif isinstance(approximator.scoring_rule, ExponentialScore):
        assert approximator.scoring_rule.is_pmp_rule is False


def test_scoring_rule_property_raises_for_multiple_rules():
    import keras
    from bayesflow.approximators import ModelComparisonApproximator
    from bayesflow.scoring_rules import CrossEntropyScore, BrierScore

    approx = ModelComparisonApproximator(
        num_models=2,
        classifier_network=keras.layers.Dense(4),
        scoring_rules={"ce": CrossEntropyScore(), "brier": BrierScore()},
    )
    with pytest.raises(AttributeError, match="Multiple scoring rules"):
        _ = approx.scoring_rule


def test_multi_rule_estimate(train_dataset, simulator):
    import keras
    from bayesflow.approximators import ModelComparisonApproximator
    from bayesflow.networks import MLP
    from bayesflow.scoring_rules import CrossEntropyScore, BrierScore
    from bayesflow import Adapter

    adapter = (
        Adapter()
        .sqrt("n")
        .broadcast("n", to="x")
        .as_set("x")
        .rename("n", "inference_conditions")
        .rename("x", "summary_variables")
        .rename("model_indices", "inference_variables")
        .drop("mu")
        .convert_dtype("float64", "float32")
    )
    from bayesflow.networks import DeepSet

    approx = ModelComparisonApproximator(
        num_models=len(simulator.simulators),
        classifier_network=MLP(widths=(8, 8)),
        summary_network=DeepSet(summary_dim=2, depth=1),
        scoring_rules={"ce": CrossEntropyScore(), "brier": BrierScore()},
        adapter=adapter,
    )
    data_shapes = keras.tree.map_structure(keras.ops.shape, train_dataset[0])
    approx.build(data_shapes)
    approx.compute_metrics(**train_dataset[0])

    num_conditions = 2
    num_models = len(simulator.simulators)
    conditions = simulator.sample(num_conditions)
    output = approx.estimate(conditions=conditions)

    assert isinstance(output, dict)
    for rule_key in ("ce", "brier"):
        assert rule_key in output
        assert "model_probs" in output[rule_key]
        assert output[rule_key]["model_probs"].shape == (num_conditions, num_models)
        assert "logits" in output[rule_key]

    assert "_summaries" in output
    assert output["_summaries"].shape[0] == num_conditions


def test_build_dataset_with_simulators_list(approximator, adapter):
    import numpy as np
    from bayesflow import make_simulator
    from bayesflow.datasets import OnlineDataset

    def prior_null():
        return dict(mu=0.0, n=4)

    def prior_alt():
        return dict(mu=np.random.normal(0, 1), n=4)

    def likelihood(mu, n):
        return dict(x=np.random.normal(mu, 1, n))

    sims = [
        make_simulator([prior_null, likelihood]),
        make_simulator([prior_alt, likelihood]),
    ]

    dataset = approximator.build_dataset(
        simulators=sims,
        adapter=adapter,
        memory_budget="20 KiB",
        num_batches=2,
    )
    assert isinstance(dataset, OnlineDataset)


def test_build_dataset_conflict_raises(approximator, simulator, adapter):
    dataset = approximator.build_dataset(
        simulator=simulator,
        adapter=adapter,
        memory_budget="20 KiB",
        num_batches=2,
    )
    with pytest.raises(ValueError, match="Exactly one"):
        approximator.build_dataset(dataset=dataset, simulator=simulator)


def test_fit_dataset_conflict_raises(approximator, train_dataset, simulator):
    approximator.compile(optimizer="AdamW")
    with pytest.raises(ValueError, match="conflicting"):
        approximator.fit(dataset=train_dataset, simulator=simulator, epochs=1)


def test_fit_with_simulators_list():
    """fit(simulators=[...]) auto-builds adapter and ModelComparisonSimulator."""
    import numpy as np
    from bayesflow import make_simulator
    from bayesflow.approximators import ModelComparisonApproximator
    from bayesflow.networks import MLP

    def prior_null():
        return dict(mu=0.0)

    def prior_alt():
        return dict(mu=np.random.normal(0, 1))

    def likelihood(mu):
        return dict(x=np.random.normal(mu, 1, 4).astype(np.float32))

    sims = [
        make_simulator([prior_null, likelihood]),
        make_simulator([prior_alt, likelihood]),
    ]

    approximator = ModelComparisonApproximator(
        num_models=2,
        classifier_network=MLP(widths=(8,)),
    )
    approximator.compile(optimizer="AdamW")
    approximator.fit(
        simulators=sims,
        inference_conditions=["x"],
        epochs=1,
        num_batches=1,
        batch_size=4,
        verbose=0,
    )
