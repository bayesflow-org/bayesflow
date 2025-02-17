# Patterns & Caveats

**Note**: This document is part of BayesFlow's developer documentation, and
aimed at people who want to extend or improve BayesFlow. For user documentation,
please refer to the examples and the public API documentation.

## Introduction

From version 2 on, BayesFlow is built on [Keras](https://keras.io/) v3, which
allows writing machine learning pipelines that run in JAX, TensorFlow and PyTorch.
By using functionality provided by Keras, and extending it with backend-specific
code where necessary, we aim to build BayesFlow in a backend-agnostic fashion as
well.

As Keras is built upon three different backend, each with different functionality
and design decisions, it has its own quirks and compromises. This documents
outlines some of them, along with the design decisions and programming patterns
we use to counter them.

This document is work in progress, so if you read through the code base and
encounter something that looks odd, but shows up in multiple places, please open
an issue so that we can add it here. Also, if you introduce a new pattern that
others will have to use in the future as well, please document it here, along
with some background information on why it is necessary and how to use it in
practice.

## Privileged `training` argument in the `call()` method cannot be passed via `kwargs`

For layers that have different behavior at training and inference time (e.g.,
dropout or batch normalization layers), a boolean `training` argument can be
exposed, see [this section of the Keras documentation](https://keras.io/guides/making_new_layers_and_models_via_subclassing/#privileged-training-argument-in-the-call-method).
If we want to pass this manually, we have to do so explicitly and not as part
of a set of keyword arguments via `**kwargs`.

@Lars: Maybe you can add more details on what is going on behind the scenes.

## Serialization

Serialization deals with the problem of storing objects to disk, and loading
them at a later point in time. This is straight-forward for data structures like
numpy arrays, but for classes with custom behavior, like approximators or neural
network layers, it is somewhat more complex.

Please refer to the Keras guide [Save, serialize, and export models](https://keras.io/guides/serialization_and_saving/)
for an introduction, and [Customizing Saving and Serialization](https://keras.io/guides/customizing_saving_and_serialization/)
for advanced concepts.

The basic idea is: by storing the arguments of the constructor of a class
(i.e., the arguments of the `__init__` function), we can later construct an
object identical to the one we have stored, except for the weights.
As the structure is identical, we can then map the stored weights to the newly
constructed object. The caveat is that all arguments have to be either basic
Python objects (like int, float, string, bool, ...) or themselves serializable.
If they are not, we have to manually specify how to serialize them, and how to
load them later on.

### Registering classes as serializable

TODO

### Serialization of custom types

In BayesFlow, we often encounter situations where we do not want to pass a
specific object (e.g., an MPL of a certain size), but we want to pass its type
(MLP) and the arguments to construct it. With the type and the arguments, we can
then construct multiple instances of the network in different places, for example
as the network inside a coupling block.

Unfortunately, `type` is not Keras serializable, so we have to serialize those
arguments manually. To complicate matters further, we also allow passing a string
instead of a type, which is then used to select the correct type.

To make it more concrete, we look at the `CouplingFlow` class, which takes the
argument `subnet` that provide the type of the subnet. It is either a
string (e.g., `"mlp"`) or a class (e.g., `bayesflow.networks.MLP`). In the first
case, we can just store the value and load it, in the latter case, we first have
to convert the type to a string that we can later convert back into a type.

We provide two helper functions that can deal with both cases:
`bayesflow.utils.serialize_value_or_type(config, name, obj)` and
`bayesflow.utils.deserialize_value_or_type(config, name)`.
In `get_config`, we use the first to store the object, whereas we use the
latter in `from_config` to load it again.

As we need all arguments to `__init__` in `get_config`, it can make sense to
build a `config` dictionary in `__init__` already, which can then be stored when
`get_config` is called. Take a look at `CouplingFlow` for an example of that.
