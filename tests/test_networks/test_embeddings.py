import pytest
import keras
import numpy as np

from bayesflow.networks.embeddings import FourierEmbedding, RecurrentEmbedding, Time2Vec, FiLM


def test_fourier_embedding_output_shape_and_type():
    embed_dim = 8
    batch_size = 4

    emb_layer = FourierEmbedding(embed_dim=embed_dim, include_identity=True)
    # use keras.ops.zeros with shape (batch_size, 1) and float32 dtype
    t = keras.ops.zeros((batch_size, 1), dtype="float32")

    emb = emb_layer(t)
    # Expected shape is (batch_size, embed_dim + 1) if include_identity else (batch_size, embed_dim)
    expected_dim = embed_dim + 1
    assert emb.shape[0] == batch_size
    assert emb.shape[1] == expected_dim
    # Check type - it should be a Keras tensor, convert to numpy for checking
    np_emb = keras.ops.convert_to_numpy(emb)
    assert np_emb.shape == (batch_size, expected_dim)


def test_fourier_embedding_without_identity():
    embed_dim = 8
    batch_size = 3

    emb_layer = FourierEmbedding(embed_dim=embed_dim, include_identity=False)
    t = keras.ops.zeros((batch_size, 1), dtype="float32")

    emb = emb_layer(t)
    expected_dim = embed_dim
    assert emb.shape[0] == batch_size
    assert emb.shape[1] == expected_dim


def test_fourier_embedding_raises_for_odd_embed_dim():
    with pytest.raises(ValueError):
        FourierEmbedding(embed_dim=7)


def test_recurrent_embedding_lstm_and_gru_shapes():
    batch_size = 2
    seq_len = 5
    dim = 3
    embed_dim = 6

    # Dummy input
    x = keras.ops.zeros((batch_size, seq_len, dim), dtype="float32")

    # lstm
    lstm_layer = RecurrentEmbedding(embed_dim=embed_dim, embedding="lstm")
    emb_lstm = lstm_layer(x)
    # Check the concatenated shape: last dimension = original dim + embed_dim
    assert emb_lstm.shape == (batch_size, seq_len, dim + embed_dim)

    # gru
    gru_layer = RecurrentEmbedding(embed_dim=embed_dim, embedding="gru")
    emb_gru = gru_layer(x)
    assert emb_gru.shape == (batch_size, seq_len, dim + embed_dim)


def test_recurrent_embedding_raises_unknown_embedding():
    with pytest.raises(ValueError):
        RecurrentEmbedding(embed_dim=4, embedding="unknown")


def test_time2vec_shapes_and_output():
    batch_size = 3
    seq_len = 7
    dim = 2
    num_periodic_features = 4

    x = keras.ops.zeros((batch_size, seq_len, dim), dtype="float32")
    time2vec_layer = Time2Vec(num_periodic_features=num_periodic_features)

    emb = time2vec_layer(x)
    # The last dimension should be dim + num_periodic_features + 1 (trend + periodic)
    expected_dim = dim + num_periodic_features + 1
    assert emb.shape == (batch_size, seq_len, expected_dim)


def test_film_modulation():
    """Test that FiLM correctly applies affine transformation."""
    batch_size = 2
    units = 4
    t_emb_dim = 8

    film_layer = FiLM(units=units, kernel_initializer="zeros")

    # Input features
    x = keras.ops.ones((batch_size, units), dtype="float32")
    # Time embedding (zeros will produce gamma=0, beta=0 with zero init)
    t_emb = keras.ops.zeros((batch_size, t_emb_dim), dtype="float32")

    output = film_layer(x, t_emb)

    # With zero-initialized Dense layer: gamma=0, beta=0
    # Expected: (1 + 0) * x + 0 = x
    np_output = keras.ops.convert_to_numpy(output)
    np_x = keras.ops.convert_to_numpy(x)
    np.testing.assert_allclose(np_output, np_x, rtol=1e-5)


def test_film_integration_with_conditional_residual():
    """Test FiLM works correctly when used in ConditionalResidual."""
    from bayesflow.networks.residual import ConditionalResidual

    width = 32
    batch_size = 4
    input_dim = 16
    cond_dim = 64

    block = ConditionalResidual(
        width=width,
        residual=True,
        activation="relu",
    )

    x = keras.ops.zeros((batch_size, input_dim), dtype="float32")
    cond = keras.ops.zeros((batch_size, cond_dim), dtype="float32")

    # Build
    block.build(((batch_size, input_dim), (batch_size, cond_dim)))

    # Check FiLM was built correctly
    assert block.film.built
    assert block.film.units == width

    # Forward pass
    output = block((x, cond))
    assert output.shape == (batch_size, width)
