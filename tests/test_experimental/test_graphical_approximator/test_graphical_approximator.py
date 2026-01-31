import pytest
from keras.src.callbacks.history import History


@pytest.mark.parametrize("mode", ["online", "offline"])
def test_graphical_approximator_single_level(mode, single_level_simulator, single_level_approximator):
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
