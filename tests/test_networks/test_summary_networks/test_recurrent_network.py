import keras

from bayesflow.networks import RecurrentNetwork

from .conftest import BATCH, SUMMARY_DIM, make_3d_input


def test_kernel_initializer_propagates():
    net = RecurrentNetwork(
        summary_dim=SUMMARY_DIM,
        hidden_dim=(8,),
        bidirectional=False,
        time_axis=None,
        time_embed_dim=4,
        dropout=0.0,
        kernel_initializer="he_normal",
    )

    y = net(make_3d_input(set_size=8), training=False)

    assert keras.ops.shape(y) == (BATCH, SUMMARY_DIM)
    assert type(net.recurrent_layers[0].kernel_initializer).__name__ == "HeNormal"
    assert type(net.output_projector.kernel_initializer).__name__ == "HeNormal"
    assert net.get_config()["kernel_initializer"] == "he_normal"
