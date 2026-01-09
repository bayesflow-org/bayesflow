# Import Layer directly from keras.layers
from keras.layers import Layer
import keras


def assert_models_equal(model1: keras.Model, model2: keras.Model):
    if not isinstance(model1, keras.Model):
        raise ValueError("model1 is not a keras.Model")
    if not isinstance(model2, keras.Model):
        raise ValueError("model2 is not a keras.Model")

    for layer1, layer2 in zip(model1._flatten_layers(), model2._flatten_layers()):
        if layer1 is model1 or layer2 is model2:
            continue
        if isinstance(layer1, keras.Model):
            assert_models_equal(layer1, layer2)
        else:
            assert_layers_equal(layer1, layer2)


def assert_layers_equal(layer1: Layer, layer2: Layer):
    msg = f"Layers {layer1.name} and {layer2.name} have different types."
    if type(layer1) is not type(layer2):
        raise ValueError(msg)

    msg = (
        f"Layers {layer1.name} and {layer2.name} have a different number of variables "
        f"({len(layer1.variables)}, {len(layer2.variables)})."
    )
    if len(layer1.variables) != len(layer2.variables):
        raise ValueError(msg)

    msg = (
        f"Layers {layer1.name} and {layer2.name} have different build status: "
        f"{layer1.built} != {layer2.built}"
    )
    if layer1.built != layer2.built:
        raise ValueError(msg)

    for v1, v2 in zip(layer1.variables, layer2.variables):
        if v1.name == "seed_generator_state":
            # keras issue: https://github.com/keras-team/keras/issues/19796
            continue

        x1 = keras.ops.convert_to_numpy(v1)
        x2 = keras.ops.convert_to_numpy(v2)
        msg = f"Variable '{v1.name}' for Layer '{layer1.name}' is not equal: {x1} != {x2}"
        if not keras.ops.all(keras.ops.isclose(x1, x2)):
            raise ValueError(msg)
