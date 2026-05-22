import keras
import pytest
from keras.src.callbacks.history import History


def test_graphical_approximator_single_level():
    from bayesflow.experimental.graphical_approximator.example_approximators import single_level_approximator
    from bayesflow.experimental.graphical_simulator.example_simulators import single_level_simulator

    single_level_simulator = single_level_simulator()
    single_level_approximator = single_level_approximator()

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

    fit = approximator.fit(dataset=data, batch_size=1, epochs=1)
    assert isinstance(fit, History)

    fit = approximator.fit(simulator=single_level_simulator, batch_size=3, num_batches=1, epochs=1)
    assert isinstance(fit, History)
    new_data = single_level_approximator.adapter(single_level_simulator.sample(3))

    samples = approximator.sample(num_samples=10, conditions=new_data)
    assert isinstance(samples, dict)

    assert approximator._batch_size_from_data(new_data) == 3
    assert isinstance(approximator._data_shapes(new_data), dict)
    assert approximator.log_prob(data)


def test_graphical_approximator_two_level():
    from bayesflow.experimental.graphical_approximator.example_approximators import two_level_approximator
    from bayesflow.experimental.graphical_simulator.example_simulators import two_level_simulator

    two_level_simulator = two_level_simulator()
    two_level_approximator = two_level_approximator()

    data = two_level_simulator.sample(1)
    adapted_data = two_level_approximator.adapter(data)
    data_shapes = two_level_approximator._data_shapes(adapted_data)

    approximator = two_level_approximator
    approximator.build(data_shapes)
    approximator.compile()

    metrics = approximator.compute_metrics(**adapted_data)
    assert isinstance(metrics, dict)
    assert "loss" in metrics.keys()

    fit = approximator.fit(dataset=data, batch_size=1, epochs=1)
    assert isinstance(fit, History)

    fit = approximator.fit(simulator=two_level_simulator, batch_size=3, num_batches=1, epochs=1)
    assert isinstance(fit, History)

    new_data = two_level_approximator.adapter(two_level_simulator.sample(3))
    samples = approximator.sample(num_samples=10, conditions=new_data)
    assert isinstance(samples, dict)

    samples_2 = approximator.sample(num_samples=3, conditions={"y": new_data["y"]})
    assert isinstance(samples_2, dict)

    assert approximator._batch_size_from_data(new_data) == 3
    assert isinstance(approximator._data_shapes(new_data), dict)
    assert approximator.log_prob(data)


def test_graphical_approximator_three_level():
    from bayesflow.experimental.graphical_approximator.example_approximators import three_level_approximator
    from bayesflow.experimental.graphical_simulator.example_simulators import three_level_simulator

    three_level_simulator = three_level_simulator()
    three_level_approximator = three_level_approximator()

    data = three_level_simulator.sample(1, meta={"N_classrooms": 10, "N_students": 10, "N_scores": 20})
    adapted_data = three_level_approximator.adapter(data)
    data_shapes = three_level_approximator._data_shapes(adapted_data)

    approximator = three_level_approximator
    approximator.build(data_shapes)
    approximator.compile()

    metrics = approximator.compute_metrics(**adapted_data)
    assert isinstance(metrics, dict)
    assert "loss" in metrics.keys()

    fit = approximator.fit(dataset=data, batch_size=1, epochs=1)
    assert isinstance(fit, History)

    fit = approximator.fit(simulator=three_level_simulator, batch_size=3, num_batches=1, epochs=1)
    assert isinstance(fit, History)

    new_data = three_level_approximator.adapter(three_level_simulator.sample(3))
    samples = approximator.sample(num_samples=10, conditions=new_data)
    assert isinstance(samples, dict)

    samples_2 = approximator.sample(num_samples=3, conditions={"y": new_data["y"]})
    assert isinstance(samples_2, dict)

    assert approximator._batch_size_from_data(new_data) == 3
    assert isinstance(approximator._data_shapes(new_data), dict)
    assert approximator.log_prob(data)


def test_graphical_approximator_crossed_design_irt():
    from bayesflow.experimental.graphical_approximator.example_approximators import crossed_design_irt_approximator
    from bayesflow.experimental.graphical_simulator.example_simulators import crossed_design_irt_simulator

    crossed_design_irt_simulator = crossed_design_irt_simulator()
    crossed_design_irt_approximator = crossed_design_irt_approximator()

    data = crossed_design_irt_simulator.sample(1, meta={"num_questions": 15, "num_students": 200})
    adapted_data = crossed_design_irt_approximator.adapter(data)
    data_shapes = crossed_design_irt_approximator._data_shapes(adapted_data)

    approximator = crossed_design_irt_approximator
    approximator.build(data_shapes)
    approximator.compile()

    metrics = approximator.compute_metrics(**adapted_data)
    assert isinstance(metrics, dict)
    assert "loss" in metrics.keys()

    fit = approximator.fit(dataset=data, batch_size=1, epochs=1)
    assert isinstance(fit, History)

    fit = approximator.fit(simulator=crossed_design_irt_simulator, batch_size=3, num_batches=1, epochs=1)
    assert isinstance(fit, History)

    new_data = crossed_design_irt_approximator.adapter(crossed_design_irt_simulator.sample(3))
    samples = approximator.sample(num_samples=10, conditions=new_data)
    assert isinstance(samples, dict)

    samples_2 = approximator.sample(num_samples=3, conditions={"obs": new_data["obs"]})
    assert isinstance(samples_2, dict)

    assert approximator._batch_size_from_data(new_data) == 3
    assert isinstance(approximator._data_shapes(new_data), dict)
    assert approximator.log_prob(data)


def test_custom_standardize():
    from bayesflow.experimental.graphical_approximator.example_approximators import crossed_design_irt_approximator
    from bayesflow.experimental.graphical_simulator.example_simulators import crossed_design_irt_simulator

    crossed_design_irt_simulator = crossed_design_irt_simulator()
    approximator = crossed_design_irt_approximator()
    approximator.compile()

    fit = approximator.fit(simulator=crossed_design_irt_simulator, batch_size=2, num_batches=1, epochs=1)
    assert isinstance(fit, History)


def test_default_adapter():
    from bayesflow.adapters import Adapter
    from bayesflow.experimental.graphical_approximator import GraphicalApproximator
    from bayesflow.experimental.graphical_approximator.example_approximators import crossed_design_irt_approximator
    from bayesflow.experimental.graphical_simulator.example_simulators import crossed_design_irt_simulator

    crossed_design_irt_simulator = crossed_design_irt_simulator()
    approximator = crossed_design_irt_approximator()
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


def test_log_prob():
    from bayesflow.adapters import Adapter
    from bayesflow.experimental.graphical_approximator import GraphicalApproximator
    from bayesflow.experimental.graphical_simulator.example_simulators import crossed_design_irt_simulator
    from bayesflow.networks import CouplingFlow, DeepSet

    crossed_design_irt_simulator = crossed_design_irt_simulator()

    adapter = Adapter()
    adapter.to_array()
    adapter.convert_dtype("float64", "float32")
    adapter.standardize(include="obs", mean=0.5, std=0.5)

    summary_networks = [
        DeepSet(summary_dim=10),
        DeepSet(summary_dim=20),
        DeepSet(summary_dim=30),
        DeepSet(summary_dim=40),
        DeepSet(summary_dim=50),
    ]
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


def test_auto_adapter():
    from bayesflow.experimental.graphical_approximator import GraphicalApproximator
    from bayesflow.experimental.graphical_simulator.example_simulators import single_level_simulator
    from bayesflow.networks import CouplingFlow, DeepSet

    simulator = single_level_simulator()
    approximator = GraphicalApproximator(
        simulator.graph.invert(),
        inference_networks=[CouplingFlow()],
        summary_networks=[DeepSet()],
    )
    assert approximator.adapter is not None


def test_build_without_data_shapes():
    from bayesflow.experimental.graphical_approximator import GraphicalApproximator
    from bayesflow.experimental.graphical_simulator.example_simulators import single_level_simulator
    from bayesflow.networks import CouplingFlow, DeepSet

    simulator = single_level_simulator()
    approximator = GraphicalApproximator(
        simulator.graph.invert(),
        inference_networks=[CouplingFlow()],
        summary_networks=[DeepSet()],
        standardize=None,  # no standardize layers so symbolic shapes don't break build()
    )
    approximator.build()
    assert approximator.built


def test_call(single_level_simulator, single_level_approximator):
    data = single_level_simulator.sample(2)
    adapted = single_level_approximator.adapter(data)
    data_shapes = single_level_approximator._data_shapes(adapted)
    single_level_approximator.build(data_shapes)

    result = single_level_approximator.call(adapted)
    assert "loss" in result


def test_wrong_summary_network_count():
    from bayesflow.experimental.graphical_approximator import GraphicalApproximator
    from bayesflow.experimental.graphical_simulator.example_simulators import crossed_design_irt_simulator
    from bayesflow.experimental.graphical_simulator.example_simulators import single_level_simulator
    from bayesflow.networks import CouplingFlow, DeepSet

    simulator = single_level_simulator()
    inverted_graph = simulator.graph.invert()

    with pytest.raises(ValueError, match="summary networks"):
        GraphicalApproximator(
            inverted_graph,
            inference_networks=[CouplingFlow()],
            summary_networks=[DeepSet(), DeepSet()],  # expects 1, got 2
        )

    simulator = crossed_design_irt_simulator()
    inverted_graph = simulator.graph.invert()

    with pytest.raises(ValueError, match="non-exchangeable"):
        GraphicalApproximator(
            inverted_graph,
            inference_networks=[CouplingFlow(), CouplingFlow(), CouplingFlow()],
            summary_networks=[DeepSet(), DeepSet(), DeepSet(), DeepSet()],  # expects 5 (4 data + 1 nonex), got 4
        )


def test_subset_data():
    from bayesflow.experimental.graphical_approximator.example_approximators import crossed_design_irt_approximator
    from bayesflow.experimental.graphical_simulator.example_simulators import crossed_design_irt_simulator

    crossed_design_irt_simulator = crossed_design_irt_simulator()
    crossed_design_irt_approximator = crossed_design_irt_approximator()

    data = crossed_design_irt_simulator.sample(2, meta={"num_questions": 15, "num_students": 200})
    assert isinstance(crossed_design_irt_approximator.subset_data(data), dict)

    data["additional_key"] = keras.random.normal((2, 1))
    with pytest.raises(KeyError):
        crossed_design_irt_approximator.subset_data(data)


@pytest.mark.tensorflow
def test_fit_with_dataset_tensorflow(single_level_simulator, single_level_approximator):
    data = single_level_simulator.sample(8)
    approximator = single_level_approximator
    approximator.compile()

    fit = approximator.fit(dataset=data, batch_size=4, epochs=1)
    assert isinstance(fit, History)


@pytest.mark.tensorflow
def test_fit_with_simulator_tensorflow(single_level_simulator, single_level_approximator):
    approximator = single_level_approximator
    approximator.compile()

    fit = approximator.fit(simulator=single_level_simulator, batch_size=4, num_batches=2, epochs=1)
    assert isinstance(fit, History)


@pytest.mark.tensorflow
def test_fit_generator_coverage(single_level_simulator, single_level_approximator):
    import tensorflow as tf

    data = single_level_simulator.sample(2)
    adapted = single_level_approximator.adapter(data)
    single_level_approximator.build(single_level_approximator._data_shapes(adapted))
    single_level_approximator.compile()

    original_from_gen = tf.data.Dataset.from_generator

    def covering_from_gen(gen_func, *args, **kwargs):
        gen = gen_func()
        for _ in range(2):
            try:
                next(gen)
            except StopIteration:
                break
        return original_from_gen(gen_func, *args, **kwargs)

    tf.data.Dataset.from_generator = covering_from_gen
    try:
        single_level_approximator.fit(dataset=data, batch_size=2, epochs=1)
        single_level_approximator.fit(simulator=single_level_simulator, batch_size=2, num_batches=1, epochs=1)
    finally:
        tf.data.Dataset.from_generator = original_from_gen
