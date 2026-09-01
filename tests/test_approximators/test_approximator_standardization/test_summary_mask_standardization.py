import numpy as np
import keras

from bayesflow import ContinuousApproximator
from bayesflow.networks import CouplingFlow, TimeSeriesTransformer


def _approximator():
    return ContinuousApproximator(
        inference_network=CouplingFlow(subnet="mlp", depth=2, subnet_kwargs=dict(widths=(16, 16))),
        summary_network=TimeSeriesTransformer(summary_dim=4, embed_dims=(8, 8), num_heads=(2, 2), dropout=0.0),
        standardize="all",
    )


def _summary_stats(approximator):
    return approximator.standardizer.standardize_layers["summary_variables"]


def test_summary_mask_reaches_standardization():
    keras.utils.set_random_seed(1)
    batch, real_max, feat, param_dim, pad_len = 2, 6, 3, 2, 4

    # per-row varying real lengths, encoded in the mask on a dense (batch, real_max, feat) tensor
    lengths = np.array([6, 3])
    real_mask = keras.ops.convert_to_tensor((np.arange(real_max)[None, :] < lengths[:, None]).astype("float32"))
    summary_variables = keras.random.normal((batch, real_max, feat), mean=10.0)
    inference_variables = keras.random.normal((batch, param_dim))

    # same data, extended with fully-masked padding
    padded_variables = keras.ops.concatenate([summary_variables, keras.ops.zeros((batch, pad_len, feat))], axis=1)
    padded_mask = keras.ops.concatenate([real_mask, keras.ops.zeros((batch, pad_len))], axis=1)

    reference = _approximator()
    reference.build_from_data(dict(inference_variables=inference_variables, summary_variables=summary_variables))
    reference.compute_metrics(
        inference_variables=inference_variables,
        summary_variables=summary_variables,
        summary_mask=real_mask,
        stage="training",
    )

    padded = _approximator()
    padded.build_from_data(dict(inference_variables=inference_variables, summary_variables=padded_variables))
    padded.compute_metrics(
        inference_variables=inference_variables,
        summary_variables=padded_variables,
        summary_mask=padded_mask,
        stage="training",
    )

    for attr in ["moving_mean", "moving_m2", "count"]:
        np.testing.assert_allclose(
            keras.ops.convert_to_numpy(getattr(_summary_stats(reference), attr)[0]),
            keras.ops.convert_to_numpy(getattr(_summary_stats(padded), attr)[0]),
        )
