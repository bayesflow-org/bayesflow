import pytest

import keras

from bayesflow.utils.serialization import serializable


@pytest.fixture(params=["coupling_flow", "flow_matching"])
def inference_network(request):
    if request.param == "coupling_flow":
        from bayesflow.networks import CouplingFlow

        return CouplingFlow(depth=2)

    elif request.param == "flow_matching":
        from bayesflow.networks import FlowMatching

        return FlowMatching(subnet_kwargs=dict(widths=(32, 32)), use_optimal_transport=False)


@pytest.fixture(params=["time_series_transformer", "fusion_transformer", "time_series_network", "custom"])
def summary_network(request):
    if request.param == "time_series_transformer":
        from bayesflow.networks import TimeSeriesTransformer

        return TimeSeriesTransformer(embed_dims=(8, 8), mlp_widths=(16, 8), mlp_depths=(1, 1))

    elif request.param == "fusion_transformer":
        from bayesflow.networks import FusionTransformer

        return FusionTransformer(
            embed_dims=(8, 8), mlp_widths=(8, 16), mlp_depths=(2, 1), template_dim=8, bidirectional=False
        )

    elif request.param == "time_series_network":
        from bayesflow.networks import TimeSeriesNetwork

        return TimeSeriesNetwork(filters=4, skip_steps=2)

    elif request.param == "custom":
        from bayesflow.networks import SummaryNetwork

        @serializable
        class Custom(SummaryNetwork):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                self.inner = keras.Sequential([keras.layers.LSTM(8), keras.layers.Dense(4)])

            def call(self, x, **kwargs):
                return self.inner(x, training=kwargs.get("stage") == "training")

        return Custom()
