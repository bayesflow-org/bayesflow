import keras


def assert_models_equal(model1: keras.Model, model2: keras.Model):
    assert isinstance(model1, keras.Model)
    assert isinstance(model2, keras.Model)

    for layer1, layer2 in zip(model1.layers, model2.layers):
        if isinstance(layer1, keras.Model):
            assert_models_equal(layer1, layer2)
        else:
            assert_layers_equal(layer1, layer2)


def assert_layers_equal(layer1: keras.Layer, layer2: keras.Layer):
    msg = (
        f"Layers {layer1.name} and {layer2.name} have a different number of variables "
        f"({len(layer1.variables)}, {len(layer2.variables)})."
    )
    assert len(layer1.variables) == len(layer2.variables), msg

    msg = f"Layers {layer1.name} and {layer2.name} have no variables."
    assert len(layer1.variables) > 0, msg

    for v1, v2 in zip(layer1.variables, layer2.variables):
        if v1.name == "seed_generator_state":
            # keras issue: https://github.com/keras-team/keras/issues/19796
            continue

        x1 = keras.ops.convert_to_numpy(v1)
        x2 = keras.ops.convert_to_numpy(v2)
        msg = f"Variable '{v1.name}' for Layer '{layer1.name}' is not equal: {x1} != {x2}"
        assert keras.ops.all(keras.ops.isclose(x1, x2)), msg

    # The names of layers need not stay the same
    # assert layer1.name == layer2.name
