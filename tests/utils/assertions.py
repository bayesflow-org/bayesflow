import keras


def assert_models_equal(model1: keras.Model, model2: keras.Model):
    assert isinstance(model1, keras.Model)
    assert isinstance(model2, keras.Model)

    for layer1, layer2 in zip(model1._flatten_layers(), model2._flatten_layers()):
        if layer1 is model1 or layer2 is model2:
            continue
        if isinstance(layer1, keras.Model):
            assert_models_equal(layer1, layer2)
        else:
            assert_layers_equal(layer1, layer2)


def assert_layers_equal(layer1: keras.Layer, layer2: keras.Layer):
    msg = f"Layers {layer1.name} and {layer2.name} have different types."
    assert type(layer1) is type(layer2), msg

    msg = (
        f"Layers {layer1.name} and {layer2.name} have a different number of variables "
        f"({len(layer1.variables)}, {len(layer2.variables)})."
    )
    assert len(layer1.variables) == len(layer2.variables), msg

    msg = f"Layers {layer1.name} and {layer2.name} have different build status: {layer1.built} != {layer2.built}"
    assert layer1.built == layer2.built, msg

    for v1, v2 in zip(layer1.variables, layer2.variables):
        if v1.name == "seed_generator_state":
            # keras issue: https://github.com/keras-team/keras/issues/19796
            continue

        x1 = keras.ops.convert_to_numpy(v1)
        x2 = keras.ops.convert_to_numpy(v2)
        msg = f"Variable '{v1.name}' for Layer '{layer1.name}' is not equal: {x1} != {x2}"
        assert keras.ops.all(keras.ops.isclose(x1, x2)), msg

    # this is turned off for now, see https://github.com/bayesflow-org/bayesflow/issues/412
    msg = f"Layers {layer1.name} and {layer2.name} have a different name."
    # assert layer1.name == layer2.name, msg
