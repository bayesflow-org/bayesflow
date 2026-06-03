"""Tests for the attention building blocks used in Set Transformer variants.

Covers:
- MultiHeadAttention (MAB)
- FFN
- SetAttention (SAB)
- InducedSetAttention (ISAB)
- PoolingByMultiHeadAttention (PMA)
"""

import keras
import pytest

from bayesflow.utils.serialization import deserialize, serialize

from tests.utils import assert_layers_equal


BATCH = 4
SET_SIZE = 8
INPUT_DIM = 16
EMBED_DIM = 32
NUM_HEADS = 4
NUM_INDUCING = 5
NUM_SEEDS = 2


@pytest.fixture()
def ffn():
    from bayesflow.networks.summary.transformers.attention.feedforward_net import FFN

    return FFN(embed_dim=EMBED_DIM)


@pytest.fixture()
def mab():
    from bayesflow.networks.summary.transformers.attention import MultiHeadAttention

    return MultiHeadAttention(embed_dim=EMBED_DIM, num_heads=NUM_HEADS)


@pytest.fixture()
def sab():
    from bayesflow.networks.summary.transformers.attention import SetAttention

    return SetAttention(embed_dim=EMBED_DIM, num_heads=NUM_HEADS)


@pytest.fixture()
def isab():
    from bayesflow.networks.summary.transformers.attention import InducedSetAttention

    return InducedSetAttention(num_inducing_points=NUM_INDUCING, embed_dim=EMBED_DIM, num_heads=NUM_HEADS)


@pytest.fixture()
def pma():
    from bayesflow.networks.summary.transformers.attention import PoolingByMultiHeadAttention

    return PoolingByMultiHeadAttention(num_seeds=NUM_SEEDS, embed_dim=EMBED_DIM, num_heads=NUM_HEADS)


@pytest.fixture()
def x():
    return keras.ops.ones((BATCH, SET_SIZE, INPUT_DIM))


@pytest.fixture()
def y():
    """Distinct key/value input for MAB tests."""
    return keras.ops.ones((BATCH, SET_SIZE + 3, INPUT_DIM))


def _serialize_roundtrip(layer):
    serialized = serialize(layer)
    deserialized = deserialize(serialized)
    reserialized = serialize(deserialized)
    return deserialized, serialized, reserialized


class TestFFN:
    def test_build_explicit(self, ffn, x):
        ffn.build(x.shape)
        assert ffn.built

    def test_build_implicit(self, ffn, x):
        ffn(x)
        assert ffn.built

    def test_output_shape(self, ffn, x):
        out = ffn(x)
        assert out.shape == (BATCH, SET_SIZE, EMBED_DIM)

    def test_compute_output_shape(self, ffn, x):
        ffn.build(x.shape)
        assert ffn.compute_output_shape(x.shape) == (BATCH, SET_SIZE, EMBED_DIM)

    def test_serialize_deserialize(self, ffn, x):
        ffn.build(x.shape)
        _, serialized, reserialized = _serialize_roundtrip(ffn)
        assert keras.tree.lists_to_tuples(serialized) == keras.tree.lists_to_tuples(reserialized)

    def test_save_load(self, tmp_path, ffn, x):
        ffn.build(x.shape)
        keras.saving.save_model(ffn, tmp_path / "ffn.keras")
        loaded = keras.saving.load_model(tmp_path / "ffn.keras")
        assert_layers_equal(ffn, loaded)


class TestMultiHeadAttention:
    def test_build_explicit(self, mab, x, y):
        mab.build(x.shape, y.shape)
        assert mab.built

    def test_build_implicit(self, mab, x, y):
        mab(x, y)
        assert mab.built

    def test_output_shape_self(self, mab, x):
        out = mab(x, x)
        assert out.shape == (BATCH, SET_SIZE, EMBED_DIM)

    def test_output_shape_cross(self, mab, x, y):
        out = mab(x, y)
        # query sequence length is preserved
        assert out.shape == (BATCH, SET_SIZE, EMBED_DIM)

    def test_compute_output_shape(self, mab, x, y):
        mab.build(x.shape, y.shape)
        assert mab.compute_output_shape(x.shape, y.shape) == (BATCH, SET_SIZE, EMBED_DIM)

    def test_variable_batch_size(self, mab, x, y):
        mab.build(x.shape, y.shape)
        for b in [1, 3]:
            xi = keras.ops.ones((b, SET_SIZE, INPUT_DIM))
            yi = keras.ops.ones((b, SET_SIZE + 3, INPUT_DIM))
            out = mab(xi, yi)
            assert out.shape[0] == b

    def test_serialize_deserialize(self, mab, x, y):
        mab.build(x.shape, y.shape)
        _, serialized, reserialized = _serialize_roundtrip(mab)
        assert keras.tree.lists_to_tuples(serialized) == keras.tree.lists_to_tuples(reserialized)

    def test_save_load(self, tmp_path, mab, x, y):
        mab.build(x.shape, y.shape)
        keras.saving.save_model(mab, tmp_path / "mab.keras")
        loaded = keras.saving.load_model(tmp_path / "mab.keras")
        assert_layers_equal(mab, loaded)

    @pytest.mark.parametrize("layer_norm", [True, False])
    def test_layer_norm_variants(self, layer_norm, x, y):
        from bayesflow.networks.summary.transformers.attention import MultiHeadAttention

        mab = MultiHeadAttention(embed_dim=EMBED_DIM, num_heads=NUM_HEADS, layer_norm=layer_norm)
        out = mab(x, y)
        assert out.shape == (BATCH, SET_SIZE, EMBED_DIM)


class TestSetAttention:
    def test_build_explicit(self, sab, x):
        sab.build(x.shape)
        assert sab.built

    def test_build_implicit(self, sab, x):
        sab(x)
        assert sab.built

    def test_output_shape(self, sab, x):
        out = sab(x)
        assert out.shape == (BATCH, SET_SIZE, EMBED_DIM)

    def test_compute_output_shape(self, sab, x):
        sab.build(x.shape)
        assert sab.compute_output_shape(x.shape) == (BATCH, SET_SIZE, EMBED_DIM)

    def test_variable_set_size(self, sab, x):
        sab.build(x.shape)
        for s in [3, 7, 20]:
            xi = keras.ops.ones((BATCH, s, INPUT_DIM))
            out = sab(xi)
            assert out.shape == (BATCH, s, EMBED_DIM)

    def test_serialize_deserialize(self, sab, x):
        sab.build(x.shape)
        _, serialized, reserialized = _serialize_roundtrip(sab)
        assert keras.tree.lists_to_tuples(serialized) == keras.tree.lists_to_tuples(reserialized)

    def test_save_load(self, tmp_path, sab, x):
        sab.build(x.shape)
        keras.saving.save_model(sab, tmp_path / "sab.keras")
        loaded = keras.saving.load_model(tmp_path / "sab.keras")
        assert_layers_equal(sab, loaded)


class TestInducedSetAttention:
    def test_build_explicit(self, isab, x):
        isab.build(x.shape)
        assert isab.built

    def test_build_implicit(self, isab, x):
        isab(x)
        assert isab.built

    def test_output_shape(self, isab, x):
        out = isab(x)
        assert out.shape == (BATCH, SET_SIZE, EMBED_DIM)

    def test_compute_output_shape(self, isab, x):
        isab.build(x.shape)
        assert isab.compute_output_shape(x.shape) == (BATCH, SET_SIZE, EMBED_DIM)

    def test_variable_set_size(self, isab, x):
        isab.build(x.shape)
        for s in [3, 7, 20]:
            xi = keras.ops.ones((BATCH, s, INPUT_DIM))
            out = isab(xi)
            assert out.shape == (BATCH, s, EMBED_DIM)

    def test_variable_batch_size(self, isab, x):
        isab.build(x.shape)
        for b in [1, 3, 8]:
            xi = keras.ops.ones((b, SET_SIZE, INPUT_DIM))
            out = isab(xi)
            assert out.shape[0] == b

    @pytest.mark.parametrize("num_inducing", [2, 4])
    def test_inducing_point_counts(self, num_inducing, x):
        from bayesflow.networks.summary.transformers.attention import InducedSetAttention

        layer = InducedSetAttention(num_inducing_points=num_inducing, embed_dim=EMBED_DIM, num_heads=NUM_HEADS)
        out = layer(x)
        assert out.shape == (BATCH, SET_SIZE, EMBED_DIM)

    def test_serialize_deserialize(self, isab, x):
        isab.build(x.shape)
        _, serialized, reserialized = _serialize_roundtrip(isab)
        assert keras.tree.lists_to_tuples(serialized) == keras.tree.lists_to_tuples(reserialized)

    def test_save_load(self, tmp_path, isab, x):
        isab.build(x.shape)
        keras.saving.save_model(isab, tmp_path / "isab.keras")
        loaded = keras.saving.load_model(tmp_path / "isab.keras")
        assert_layers_equal(isab, loaded)


class TestPoolingByMultiHeadAttention:
    def test_build_explicit(self, pma, x):
        pma.build(x.shape)
        assert pma.built

    def test_build_implicit(self, pma, x):
        pma(x)
        assert pma.built

    def test_output_shape(self, pma, x):
        out = pma(x)
        assert out.shape == (BATCH, NUM_SEEDS * EMBED_DIM)

    def test_compute_output_shape(self, pma, x):
        pma.build(x.shape)
        assert pma.compute_output_shape(x.shape) == (BATCH, NUM_SEEDS * EMBED_DIM)

    def test_variable_set_size(self, pma, x):
        pma.build(x.shape)
        for s in [1, 3]:
            xi = keras.ops.ones((BATCH, s, INPUT_DIM))
            out = pma(xi)
            assert out.shape == (BATCH, NUM_SEEDS * EMBED_DIM)

    @pytest.mark.parametrize("num_seeds", [1, 2])
    def test_num_seeds(self, num_seeds, x):
        from bayesflow.networks.summary.transformers.attention import PoolingByMultiHeadAttention

        layer = PoolingByMultiHeadAttention(num_seeds=num_seeds, embed_dim=EMBED_DIM, num_heads=NUM_HEADS)
        out = layer(x)
        assert out.shape == (BATCH, num_seeds * EMBED_DIM)

    def test_serialize_deserialize(self, pma, x):
        pma.build(x.shape)
        _, serialized, reserialized = _serialize_roundtrip(pma)
        assert keras.tree.lists_to_tuples(serialized) == keras.tree.lists_to_tuples(reserialized)

    def test_save_load(self, tmp_path, pma, x):
        pma.build(x.shape)
        keras.saving.save_model(pma, tmp_path / "pma.keras")
        loaded = keras.saving.load_model(tmp_path / "pma.keras")
        assert_layers_equal(pma, loaded)
