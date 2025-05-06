import pytest
import keras

from bayesflow.networks.embeddings import (
    FourierEmbedding,
    RecurrentEmbedding,
    Time2Vec,
)


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
