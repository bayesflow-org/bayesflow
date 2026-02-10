import keras
import pytest
from keras.src.callbacks.history import History


@pytest.mark.parametrize("mode", ["online", "offline"])
def test_graphical_approximator_single_level(mode, single_level_simulator, single_level_approximator):
    # fitting models with dynamic output shapes fails in tensorflow
    data = single_level_simulator.sample(1)
    adapted_data = single_level_approximator.adapter(data)
    data_shapes = single_level_approximator._data_shapes(adapted_data)

    approximator = single_level_approximator
    approximator.build(data_shapes)
    approximator.compile()

    metrics = approximator.compute_metrics(**adapted_data)
    assert isinstance(metrics, dict)
    assert "loss" in metrics.keys()

    if mode == "online":
        fit = approximator.fit(simulator=single_level_simulator, batch_size=3, num_batches=1, epochs=1)
    else:
        fit = approximator.fit(dataset=data, batch_size=1, epochs=1)
    assert isinstance(fit, History)

    new_data = single_level_approximator.adapter(single_level_simulator.sample(3))

    samples = approximator.sample(num_samples=10, conditions=new_data)
    assert isinstance(samples, dict)

    assert approximator._batch_size_from_data(new_data) == 3
    assert isinstance(approximator._data_shapes(new_data), dict)
    assert approximator.log_prob(data)


@pytest.mark.parametrize("mode", ["online", "offline"])
def test_graphical_approximator_two_level(mode, two_level_simulator, two_level_approximator):
    data = two_level_simulator.sample(1)
    adapted_data = two_level_approximator.adapter(data)
    data_shapes = two_level_approximator._data_shapes(adapted_data)

    approximator = two_level_approximator
    approximator.build(data_shapes)
    approximator.compile()

    metrics = approximator.compute_metrics(**adapted_data)
    assert isinstance(metrics, dict)
    assert "loss" in metrics.keys()

    if mode == "online":
        fit = approximator.fit(simulator=two_level_simulator, batch_size=3, num_batches=1, epochs=1)
    else:
        fit = approximator.fit(dataset=data, batch_size=1, epochs=1)
    assert isinstance(fit, History)

    new_data = two_level_approximator.adapter(two_level_simulator.sample(3))
    samples = approximator.sample(num_samples=10, conditions=new_data)
    assert isinstance(samples, dict)

    samples_2 = approximator.sample(num_samples=3, conditions={"y": new_data["y"]})
    assert isinstance(samples_2, dict)

    assert approximator._batch_size_from_data(new_data) == 3
    assert isinstance(approximator._data_shapes(new_data), dict)
    assert approximator.log_prob(data)


@pytest.mark.parametrize("mode", ["online", "offline"])
def test_graphical_approximator_two_level_repeated_roots(
    mode, two_level_repeated_roots_simulator, two_level_repeated_roots_approximator
):
    data = two_level_repeated_roots_simulator.sample(1)
    adapted_data = two_level_repeated_roots_approximator.adapter(data)
    data_shapes = two_level_repeated_roots_approximator._data_shapes(adapted_data)

    approximator = two_level_repeated_roots_approximator
    approximator.build(data_shapes)
    approximator.compile()

    metrics = approximator.compute_metrics(**adapted_data)
    assert isinstance(metrics, dict)
    assert "loss" in metrics.keys()

    if mode == "online":
        fit = approximator.fit(simulator=two_level_repeated_roots_simulator, batch_size=3, num_batches=1, epochs=1)
    else:
        fit = approximator.fit(dataset=data, batch_size=1, epochs=1)
    assert isinstance(fit, History)

    new_data = two_level_repeated_roots_approximator.adapter(two_level_repeated_roots_simulator.sample(3))
    samples = approximator.sample(num_samples=10, conditions=new_data)
    assert isinstance(samples, dict)

    samples_2 = approximator.sample(num_samples=3, conditions={"y": new_data["y"]})
    assert isinstance(samples_2, dict)

    assert approximator._batch_size_from_data(new_data) == 3
    assert isinstance(approximator._data_shapes(new_data), dict)
    assert approximator.log_prob(data)


@pytest.mark.parametrize("mode", ["online", "offline"])
def test_graphical_approximator_three_level(mode, three_level_simulator, three_level_approximator):
    data = three_level_simulator.sample(1, meta={"N_classrooms": 10, "N_students": 10, "N_scores": 20})
    adapted_data = three_level_approximator.adapter(data)
    data_shapes = three_level_approximator._data_shapes(adapted_data)

    approximator = three_level_approximator
    approximator.build(data_shapes)
    approximator.compile()

    metrics = approximator.compute_metrics(**adapted_data)
    assert isinstance(metrics, dict)
    assert "loss" in metrics.keys()

    if mode == "online":
        fit = approximator.fit(simulator=three_level_simulator, batch_size=3, num_batches=1, epochs=1)
    else:
        fit = approximator.fit(dataset=data, batch_size=1, epochs=1)
    assert isinstance(fit, History)

    new_data = three_level_approximator.adapter(three_level_simulator.sample(3))
    samples = approximator.sample(num_samples=10, conditions=new_data)
    assert isinstance(samples, dict)

    samples_2 = approximator.sample(num_samples=3, conditions={"y": new_data["y"]})
    assert isinstance(samples_2, dict)

    assert approximator._batch_size_from_data(new_data) == 3
    assert isinstance(approximator._data_shapes(new_data), dict)
    assert approximator.log_prob(data)


@pytest.mark.parametrize("mode", ["online", "offline"])
def test_graphical_approximator_crossed_design_irt(mode, crossed_design_irt_simulator, crossed_design_irt_approximator):
    data = crossed_design_irt_simulator.sample(1, meta={"num_questions": 15, "num_students": 200})
    adapted_data = crossed_design_irt_approximator.adapter(data)
    data_shapes = crossed_design_irt_approximator._data_shapes(adapted_data)

    approximator = crossed_design_irt_approximator
    approximator.build(data_shapes)
    approximator.compile()

    metrics = approximator.compute_metrics(**adapted_data)
    assert isinstance(metrics, dict)
    assert "loss" in metrics.keys()

    if mode == "online":
        fit = approximator.fit(simulator=crossed_design_irt_simulator, batch_size=3, num_batches=1, epochs=1)
    else:
        fit = approximator.fit(dataset=data, batch_size=1, epochs=1)
    assert isinstance(fit, History)

    new_data = crossed_design_irt_approximator.adapter(crossed_design_irt_simulator.sample(3))
    samples = approximator.sample(num_samples=10, conditions=new_data)
    assert isinstance(samples, dict)

    samples_2 = approximator.sample(num_samples=3, conditions={"obs": new_data["obs"]})
    assert isinstance(samples_2, dict)

    assert approximator._batch_size_from_data(new_data) == 3
    assert isinstance(approximator._data_shapes(new_data), dict)
    assert approximator.log_prob(data)


def test_custom_standardize(crossed_design_irt_simulator, crossed_design_irt_approximator):
    from bayesflow.adapters import Adapter
    from bayesflow.experimental.graphical_approximator import GraphicalApproximator
    from bayesflow.networks import CouplingFlow, DeepSet

    adapter = Adapter()
    adapter.to_array()
    adapter.convert_dtype("float64", "float32")

    summary_networks = [DeepSet(summary_dim=10), DeepSet(summary_dim=20)]
    inference_networks = [CouplingFlow(), CouplingFlow(), CouplingFlow()]

    inverted_graph = crossed_design_irt_simulator.graph.invert()
    approximator = GraphicalApproximator(
        inverted_graph,
        adapter=adapter,
        inference_networks=inference_networks,
        summary_networks=summary_networks,
        standardize="question_mean",
    )

    data = crossed_design_irt_simulator.sample(1, meta={"num_questions": 15, "num_students": 200})
    adapted_data = crossed_design_irt_approximator.adapter(data)
    data_shapes = crossed_design_irt_approximator._data_shapes(adapted_data)

    approximator = crossed_design_irt_approximator
    approximator.build(data_shapes)
    approximator.compile()

    fit = approximator.fit(simulator=crossed_design_irt_simulator, batch_size=2, num_batches=1, epochs=1)
    assert isinstance(fit, History)


def test_default_adapter(crossed_design_irt_simulator, crossed_design_irt_approximator):
    from bayesflow.adapters import Adapter
    from bayesflow.experimental.graphical_approximator import GraphicalApproximator
    from bayesflow.networks import CouplingFlow, DeepSet

    summary_networks = [DeepSet(summary_dim=10), DeepSet(summary_dim=20)]
    inference_networks = [CouplingFlow(), CouplingFlow(), CouplingFlow()]

    inverted_graph = crossed_design_irt_simulator.graph.invert()
    approximator = GraphicalApproximator(
        inverted_graph,
        inference_networks=inference_networks,
        summary_networks=summary_networks,
        standardize="question_mean",
    )

    data = crossed_design_irt_simulator.sample(1, meta={"num_questions": 15, "num_students": 200})
    adapted_data = crossed_design_irt_approximator.adapter(data)
    data_shapes = crossed_design_irt_approximator._data_shapes(adapted_data)

    approximator = crossed_design_irt_approximator
    approximator.build(data_shapes)
    approximator.compile()

    fit = approximator.fit(simulator=crossed_design_irt_simulator, batch_size=2, num_batches=1, epochs=1)
    assert isinstance(fit, History)

    assert isinstance(GraphicalApproximator.build_adapter(), Adapter)


@pytest.mark.parametrize(
    ("simulator", "approximator"),
    [
        ("single_level_simulator", "single_level_approximator"),
        ("two_level_simulator", "two_level_approximator"),
        ("three_level_simulator", "three_level_approximator"),
        ("crossed_design_irt_simulator", "crossed_design_irt_approximator"),
    ],
)
def test_serialization(simulator, approximator, request):
    from bayesflow.experimental.graphical_approximator import GraphicalApproximator

    simulator = request.getfixturevalue(simulator)
    approximator = request.getfixturevalue(approximator)

    config = approximator.get_config()
    assert isinstance(GraphicalApproximator.from_config(config), GraphicalApproximator)


def test_log_prob(crossed_design_irt_simulator):
    from bayesflow.adapters import Adapter
    from bayesflow.experimental.graphical_approximator import GraphicalApproximator
    from bayesflow.networks import CouplingFlow, DeepSet

    adapter = Adapter()
    adapter.to_array()
    adapter.convert_dtype("float64", "float32")
    adapter.log(["question_mean"])

    summary_networks = [DeepSet(summary_dim=10), DeepSet(summary_dim=20)]
    inference_networks = [CouplingFlow(), CouplingFlow(), CouplingFlow()]

    inverted_graph = crossed_design_irt_simulator.graph.invert()
    approximator = GraphicalApproximator(
        inverted_graph,
        adapter=adapter,
        inference_networks=inference_networks,
        summary_networks=summary_networks,
        standardize="question_mean",
    )

    data = crossed_design_irt_simulator.sample(2)
    data_shapes = approximator._data_shapes(data)
    approximator.build(data_shapes)
    approximator.compile()

    assert approximator.log_prob(data) is not None


def test_subset_data(crossed_design_irt_simulator, crossed_design_irt_approximator):
    data = crossed_design_irt_simulator.sample(2, meta={"num_questions": 15, "num_students": 200})
    assert isinstance(crossed_design_irt_approximator.subset_data(data), dict)

    data["additional_key"] = keras.random.normal((2, 1))
    with pytest.raises(KeyError):
        crossed_design_irt_approximator.subset_data(data)
