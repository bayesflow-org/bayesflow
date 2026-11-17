import keras
import io
import pytest
from contextlib import redirect_stdout

from tests.utils import assert_models_equal


def first_tensor_batch(dataset):
    return keras.tree.map_structure(keras.ops.convert_to_tensor, dataset[0])


def test_build(estimator, train_dataset):
    assert estimator.built is False

    data_shapes = keras.tree.map_structure(keras.ops.shape, train_dataset[0])
    estimator.build(data_shapes)

    assert estimator.built is True
    assert estimator.inference_network.built is True
    if estimator.summary_network is not None:
        assert estimator.summary_network.built is True


def test_all_standardization_does_not_standardize_model_indices(
    adapter, classifier_network, summary_network, train_dataset
):
    from bayesflow.estimators import ModelComparisonEstimator
    from bayesflow.networks import ScoringRuleNetwork
    from bayesflow.scoring_rules import CrossEntropyScore

    estimator = ModelComparisonEstimator(
        inference_network=ScoringRuleNetwork(
            scoring_rules={"scoring_rule": CrossEntropyScore()}, subnet=classifier_network
        ),
        adapter=adapter,
        summary_network=summary_network,
        standardize="all",
    )
    batch = first_tensor_batch(train_dataset)
    data_shapes = keras.tree.map_structure(keras.ops.shape, batch)

    estimator.build(data_shapes)
    assert "inference_variables" not in estimator.standardizer.standardize

    model_indices = batch["inference_variables"]
    standardized_indices = estimator.standardizer.maybe_standardize(
        model_indices, key="inference_variables", stage="training"
    )
    assert keras.ops.allclose(standardized_indices, model_indices)

    metrics = estimator.compute_metrics(**batch)
    assert keras.ops.isfinite(metrics["loss"])
    assert keras.ops.convert_to_numpy(metrics["loss"]) >= 0.0


def test_build_adapter():
    from bayesflow.estimators import ModelComparisonEstimator

    _ = ModelComparisonEstimator.build_adapter(
        inference_conditions=["foo", "bar"],
        summary_variables=["observables"],
        inference_variables=["indices"],
    )


def test_build_dataset(estimator, simulator, adapter):
    from bayesflow.datasets import OnlineDataset

    dataset = estimator.build_dataset(
        simulator=simulator,
        adapter=adapter,
        memory_budget="20 KiB",
        num_batches=2,
    )
    assert isinstance(dataset, OnlineDataset)


def test_fit(estimator, train_dataset, validation_dataset):
    estimator.compile(optimizer="AdamW")
    num_epochs = 1

    # Capture ostream and train model
    with io.StringIO() as stream:
        with redirect_stdout(stream):
            estimator.fit(dataset=train_dataset, validation_data=validation_dataset, epochs=num_epochs)

        output = stream.getvalue()
    # check that the loss is shown
    assert "loss" in output


def test_save_and_load(tmp_path, estimator, train_dataset, validation_dataset):
    # to save, the model must be built
    batch = first_tensor_batch(train_dataset)
    data_shapes = keras.tree.map_structure(keras.ops.shape, batch)
    estimator.build(data_shapes)
    estimator.compute_metrics(**batch)

    keras.saving.save_model(estimator, tmp_path / "model.keras")
    loaded = keras.saving.load_model(tmp_path / "model.keras")

    assert_models_equal(estimator, loaded)


def test_estimate(estimator, train_dataset, simulator):
    batch = first_tensor_batch(train_dataset)
    data_shapes = keras.tree.map_structure(keras.ops.shape, batch)
    estimator.build(data_shapes)
    estimator.compute_metrics(**batch)

    num_conditions = 2
    num_models = len(simulator.simulators)
    conditions = simulator.sample(num_conditions)
    output = estimator.estimate(conditions=conditions)

    assert isinstance(output, dict)
    assert "model_probs" in output
    assert output["model_probs"].shape == (num_conditions, num_models)

    assert "log_odds" in output
    assert output["log_odds"].shape == (num_conditions, num_models)
    assert "logits" not in output
    assert "log_bayes_factors" not in output  # no model_prior -> Bayes factors not computed

    assert "_summaries" not in output

    if estimator.summary_network is not None:
        output_with_summaries = estimator.estimate(conditions=conditions, return_summaries=True)
        assert "_summaries" in output_with_summaries
        assert output_with_summaries["_summaries"].ndim == 2
        assert output_with_summaries["_summaries"].shape[0] == num_conditions


def test_estimate_log_bayes_factors_prior(estimator, train_dataset, simulator):
    """log_bayes_factors removes the model-prior log-odds; absent when no prior is given."""
    import numpy as np

    batch = first_tensor_batch(train_dataset)
    data_shapes = keras.tree.map_structure(keras.ops.shape, batch)
    estimator.build(data_shapes)
    estimator.compute_metrics(**batch)

    num_models = len(simulator.simulators)
    conditions = simulator.sample(2)

    # no prior -> Bayes factors are not computed
    out = estimator.estimate(conditions=conditions)
    assert "log_bayes_factors" not in out

    # non-uniform prior shifts log Bayes factors by the prior log-odds
    prior = np.linspace(1.0, 2.0, num_models)
    prior = prior / prior.sum()
    out = estimator.estimate(conditions=conditions, model_prior=prior)
    log_prior_odds = np.log(prior) - np.log(prior[0])
    assert np.allclose(out["log_bayes_factors"], out["log_odds"] - log_prior_odds, atol=1e-6)


def test_multi_rule_estimate(train_dataset, simulator):
    import keras
    from bayesflow.estimators import ModelComparisonEstimator
    from bayesflow.networks import MLP, ScoringRuleNetwork
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

    estimator = ModelComparisonEstimator(
        inference_network=ScoringRuleNetwork(
            scoring_rules={"ce": CrossEntropyScore(), "brier": BrierScore()},
            subnet=MLP(widths=(8, 8)),
        ),
        summary_network=DeepSet(summary_dim=2, depth=1),
        adapter=adapter,
    )
    batch = first_tensor_batch(train_dataset)
    data_shapes = keras.tree.map_structure(keras.ops.shape, batch)
    estimator.build(data_shapes)
    estimator.compute_metrics(**batch)

    import numpy as np

    num_conditions = 2
    num_models = len(simulator.simulators)
    conditions = simulator.sample(num_conditions)

    # merge_scores=False -> nested dict keyed by rule name, each with the single-rule structure
    output = estimator.estimate(conditions=conditions, merge_scores=False)

    assert isinstance(output, dict)
    for rule_key in ("ce", "brier"):
        assert rule_key in output
        assert "model_probs" in output[rule_key]
        assert output[rule_key]["model_probs"].shape == (num_conditions, num_models)
        assert "log_odds" in output[rule_key]

    assert "_summaries" not in output

    # merge_scores=True (default) -> flat dict, log-pooled over log_odds
    merged = estimator.estimate(conditions=conditions)
    assert set(merged) == {"model_probs", "log_odds"}
    assert merged["log_odds"].shape == (num_conditions, num_models)
    expected_log_odds = np.mean([output[rule_key]["log_odds"] for rule_key in ("ce", "brier")], axis=0)
    assert np.allclose(merged["log_odds"], expected_log_odds, atol=1e-5)

    # return_summaries adds a top-level _summaries key (here checked in nested mode)
    output_with_summaries = estimator.estimate(conditions=conditions, merge_scores=False, return_summaries=True)
    assert "_summaries" in output_with_summaries
    assert output_with_summaries["_summaries"].shape[0] == num_conditions


def test_merge_rule_estimates_log_pool():
    """Rules pool by a logarithmic opinion pool: the mean of the length-M log_odds."""
    import numpy as np
    from bayesflow.estimators import ModelComparisonEstimator

    log_odds_a = np.array([[0.0, 2.0, 0.5], [0.0, -1.0, 1.0]])
    log_odds_b = np.array([[0.0, -0.5, 1.5], [0.0, 2.0, -1.0]])

    merged = ModelComparisonEstimator._merge_rule_estimates(
        {"a": {"log_odds": log_odds_a}, "b": {"log_odds": log_odds_b}}
    )

    assert set(merged) == {"log_odds", "model_probs"}
    expected = np.mean([log_odds_a, log_odds_b], axis=0)
    assert np.allclose(merged["log_odds"], expected, atol=1e-6)
    exp = np.exp(expected - expected.max(-1, keepdims=True))
    assert np.allclose(merged["model_probs"], exp / exp.sum(-1, keepdims=True), atol=1e-6)


def test_build_dataset_with_simulators_list(estimator, adapter):
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

    dataset = estimator.build_dataset(
        simulators=sims,
        adapter=adapter,
        memory_budget="20 KiB",
        num_batches=2,
    )
    assert isinstance(dataset, OnlineDataset)


def test_build_dataset_conflict_raises(estimator, simulator, adapter):
    dataset = estimator.build_dataset(
        simulator=simulator,
        adapter=adapter,
        memory_budget="20 KiB",
        num_batches=2,
    )
    with pytest.raises(ValueError, match="Exactly one"):
        estimator.build_dataset(dataset=dataset, simulator=simulator)


def test_fit_dataset_conflict_raises(estimator, train_dataset, simulator):
    estimator.compile(optimizer="AdamW")
    with pytest.raises(ValueError, match="conflicting"):
        estimator.fit(dataset=train_dataset, simulator=simulator, epochs=1)


def test_fit_with_single_simulator():
    """fit(simulator=ModelComparisonSimulator(...)) takes the single-simulator fast path."""
    import numpy as np
    from bayesflow import make_simulator
    from bayesflow.estimators import ModelComparisonEstimator
    from bayesflow.networks import MLP, ScoringRuleNetwork
    from bayesflow.scoring_rules import CrossEntropyScore
    from bayesflow.simulators import ModelComparisonSimulator

    def prior_null():
        return dict(mu=0.0)

    def prior_alt():
        return dict(mu=np.random.normal(0, 1))

    def likelihood(mu):
        return dict(x=np.random.normal(mu, 1, 4).astype(np.float32))

    mc_simulator = ModelComparisonSimulator(
        simulators=[make_simulator([prior_null, likelihood]), make_simulator([prior_alt, likelihood])]
    )

    estimator = ModelComparisonEstimator(
        inference_network=ScoringRuleNetwork(
            scoring_rules={"scoring_rule": CrossEntropyScore()}, subnet=MLP(widths=(8,))
        )
    )
    estimator.compile(optimizer="AdamW")
    estimator.fit(
        simulator=mc_simulator,
        inference_conditions=["x"],
        epochs=1,
        num_batches=1,
        batch_size=4,
        verbose=0,
    )


def test_log_prob_raises(estimator, train_dataset):
    """log_prob() is not supported and raises NotImplementedError."""
    data_shapes = keras.tree.map_structure(keras.ops.shape, train_dataset[0])
    estimator.build(data_shapes)

    with pytest.raises(NotImplementedError, match="log_prob"):
        estimator.log_prob()


def test_fit_with_simulators_list():
    """fit(simulators=[...]) auto-builds adapter and ModelComparisonSimulator."""
    import numpy as np
    from bayesflow import make_simulator
    from bayesflow.estimators import ModelComparisonEstimator
    from bayesflow.networks import MLP, ScoringRuleNetwork
    from bayesflow.scoring_rules import CrossEntropyScore

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

    estimator = ModelComparisonEstimator(
        inference_network=ScoringRuleNetwork(
            scoring_rules={"scoring_rule": CrossEntropyScore()}, subnet=MLP(widths=(8,))
        ),
    )
    estimator.compile(optimizer="AdamW")
    estimator.fit(
        simulators=sims,
        inference_conditions=["x"],
        epochs=1,
        num_batches=1,
        batch_size=4,
        verbose=0,
    )
