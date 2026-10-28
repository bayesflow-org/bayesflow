# Potential Pitfalls

This document covers things we have learned during development that might cause problems or hard to find bugs.

## Privileged `training` argument in the `call()` method cannot be passed via `kwargs`

For layers that have different behavior at training and inference time (e.g.,
dropout or batch normalization layers), a boolean `training` argument can be
exposed, see [this section of the Keras documentation](https://keras.io/guides/making_new_layers_and_models_via_subclassing/#privileged-training-argument-in-the-call-method).
If we want to pass this manually, we have to do so explicitly and not as part
of a set of keyword arguments via `**kwargs`.

@Lars: Maybe you can add more details on what is going on behind the scenes.
