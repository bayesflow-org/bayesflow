# from keras.src.callbacks.history import History
#
#
# def test_graphical_approximator_single_level(single_level_simulator, single_level_approximator):
#     data = single_level_approximator.adapter(single_level_simulator.sample(2))
#     data_shapes = single_level_approximator._data_shapes(data)
#
#     approximator = single_level_approximator
#     approximator.build(data_shapes)
#     approximator.compile()
#
#     metrics = approximator.compute_metrics(**data)
#     assert isinstance(metrics, dict)
#     assert "loss" in metrics.keys()
#
#     fit = approximator.fit(dataset=data, epochs=1)
#     assert isinstance(fit, History)
#
#     new_data = single_level_approximator.adapter(single_level_simulator.sample(3))
#     samples = approximator.sample(num_samples=10, conditions=new_data)
#     assert isinstance(samples, dict)
#
#     samples_2 = approximator.sample(num_samples=3, conditions={"x": new_data["x"], "y": new_data["y"]})
#     assert isinstance(samples_2, dict)
#
#     assert approximator._batch_size_from_data(new_data) == 3
#     assert isinstance(approximator._data_shapes(new_data), dict)
#
#
# def test_graphical_approximator_two_level(two_level_simulator, two_level_approximator):
#     data = two_level_approximator.adapter(two_level_simulator.sample(2))
#     data_shapes = two_level_approximator._data_shapes(data)
#
#     approximator = two_level_approximator
#     approximator.build(data_shapes)
#     approximator.compile()
#
#     metrics = approximator.compute_metrics(**data)
#     assert isinstance(metrics, dict)
#     assert "loss" in metrics.keys()
#
#     fit = approximator.fit(dataset=data, epochs=1)
#     assert isinstance(fit, History)
#
#     new_data = two_level_approximator.adapter(two_level_simulator.sample(3))
#     samples = approximator.sample(num_samples=10, conditions=new_data)
#     assert isinstance(samples, dict)
#
#     samples_2 = approximator.sample(num_samples=3, conditions={"y": new_data["y"]})
#     assert isinstance(samples_2, dict)
#
#     assert approximator._batch_size_from_data(new_data) == 3
#     assert isinstance(approximator._data_shapes(new_data), dict)
#
#
# def test_graphical_approximator_three_level(three_level_simulator, three_level_approximator):
#     data = three_level_approximator.adapter(three_level_simulator.sample(2))
#     data_shapes = three_level_approximator._data_shapes(data)
#
#     approximator = three_level_approximator
#     approximator.build(data_shapes)
#     approximator.compile()
#
#     metrics = approximator.compute_metrics(**data)
#     assert isinstance(metrics, dict)
#     assert "loss" in metrics.keys()
#
#     fit = approximator.fit(dataset=data, epochs=1)
#     assert isinstance(fit, History)
#
#     new_data = three_level_approximator.adapter(three_level_simulator.sample(3))
#     samples = approximator.sample(num_samples=10, conditions=new_data)
#     assert isinstance(samples, dict)
#
#     samples_2 = approximator.sample(num_samples=3, conditions={"y": new_data["y"]})
#     assert isinstance(samples_2, dict)
#
#     assert approximator._batch_size_from_data(new_data) == 3
#     assert isinstance(approximator._data_shapes(new_data), dict)
#
#
# def test_graphical_approximator_crossed_design_irt(crossed_design_irt_simulator, crossed_design_irt_approximator):
#     data = crossed_design_irt_approximator.adapter(crossed_design_irt_simulator.sample(2))
#     data_shapes = crossed_design_irt_approximator._data_shapes(data)
#
#     approximator = crossed_design_irt_approximator
#     approximator.build(data_shapes)
#     approximator.compile()
#
#     metrics = approximator.compute_metrics(**data)
#     assert isinstance(metrics, dict)
#     assert "loss" in metrics.keys()
#
#     fit = approximator.fit(dataset=data, epochs=1)
#     assert isinstance(fit, History)
#
#     new_data = crossed_design_irt_approximator.adapter(crossed_design_irt_simulator.sample(3))
#     samples = approximator.sample(num_samples=10, conditions=new_data)
#     assert isinstance(samples, dict)
#
#     samples_2 = approximator.sample(num_samples=3, conditions={"obs": new_data["obs"]})
#     assert isinstance(samples_2, dict)
#
#     assert approximator._batch_size_from_data(new_data) == 3
#     assert isinstance(approximator._data_shapes(new_data), dict)
