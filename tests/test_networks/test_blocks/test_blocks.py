"""Tests for DenseBlock, ConditionalDenseBlock and DiffusionTransformerBlock."""

import keras
import pytest

from bayesflow.networks.helpers import DenseBlock, ConditionalDenseBlock
from bayesflow.utils.serialization import deserialize, serialize

from ...utils import assert_layers_equal


class TestDenseBlock:
    def test_output_shape(self, dense_block, dense_build_shapes):
        dense_block.build(**dense_build_shapes)
        x = keras.random.normal((4, 6))
        y = dense_block(x, training=False)
        assert y.shape == (4, 8)

    def test_output_shape_no_residual(self, dense_build_shapes):
        block = DenseBlock(width=8, residual=False)
        block.build(**dense_build_shapes)
        x = keras.random.normal((4, 6))
        y = block(x, training=False)
        assert y.shape == (4, 8)

    @pytest.mark.parametrize("norm", [None, "rms"])
    def test_output_shape_norms(self, norm, dense_build_shapes):
        block = DenseBlock(width=8, norm=norm)
        block.build(**dense_build_shapes)
        x = keras.random.normal((4, 6))
        y = block(x, training=False)
        assert y.shape == (4, 8)

    def test_output_changes_with_training(self, dense_build_shapes):
        """Dropout should make train/eval outputs differ (in expectation)."""
        block = DenseBlock(width=8, dropout=0.5)
        block.build(**dense_build_shapes)
        x = keras.random.normal((64, 6))
        y_train = block(x, training=True)
        y_eval = block(x, training=False)
        # They might coincidentally be equal, but almost never for 64 samples
        assert not keras.ops.all(keras.ops.isclose(y_train, y_eval))

    def test_serialize_deserialize(self, dense_block, dense_build_shapes):
        dense_block.build(**dense_build_shapes)

        serialized = serialize(dense_block)
        deserialized = deserialize(serialized)
        reserialized = serialize(deserialized)

        assert keras.tree.lists_to_tuples(serialized) == keras.tree.lists_to_tuples(reserialized)

    def test_save_and_load(self, tmp_path, dense_block, dense_build_shapes):
        dense_block.build(**dense_build_shapes)

        keras.saving.save_model(dense_block, tmp_path / "model.keras")
        loaded = keras.saving.load_model(tmp_path / "model.keras")

        assert_layers_equal(dense_block, loaded)

    def test_output_fidelity_after_load(self, tmp_path, dense_block, dense_build_shapes):
        dense_block.build(**dense_build_shapes)
        x = keras.random.normal((4, 6))
        y_before = dense_block(x, training=False)

        keras.saving.save_model(dense_block, tmp_path / "model.keras")
        loaded = keras.saving.load_model(tmp_path / "model.keras")
        y_after = loaded(x, training=False)

        assert keras.ops.all(keras.ops.isclose(y_before, y_after))


class TestConditioningBlock:
    """Shared behaviour for blocks taking a ``(x, conditioning)`` tuple.

    Parametrized via the ``conditioning_block`` fixture over ConditionalDenseBlock
    and DiffusionTransformerBlock.
    """

    def test_output_shape(self, conditioning_block):
        block = conditioning_block["make"]()
        block.build(**conditioning_block["build_shapes"])
        x = keras.random.normal(conditioning_block["x_shape"])
        cond = keras.random.normal(conditioning_block["cond_shape"])
        y = block((x, cond), training=False)
        assert y.shape == conditioning_block["out_shape"]

    def test_output_changes_with_training(self, conditioning_block):
        """Dropout should make train/eval outputs differ (in expectation)."""
        block = conditioning_block["make"](dropout=0.5)
        block.build(**conditioning_block["build_shapes"])
        x = keras.random.normal(conditioning_block["x_shape"])
        cond = keras.random.normal(conditioning_block["cond_shape"])
        y_train = block((x, cond), training=True)
        y_eval = block((x, cond), training=False)
        assert not keras.ops.all(keras.ops.isclose(y_train, y_eval))

    def test_conditioning_changes_output(self, conditioning_block):
        """Different conditioning vectors must produce different outputs."""
        block = conditioning_block["make"]()
        block.build(**conditioning_block["build_shapes"])
        x = keras.random.normal(conditioning_block["x_shape"])
        cond_a = keras.random.normal(conditioning_block["cond_shape"])
        cond_b = keras.random.normal(conditioning_block["cond_shape"])
        y_a = block((x, cond_a), training=False)
        y_b = block((x, cond_b), training=False)
        assert not keras.ops.all(keras.ops.isclose(y_a, y_b))

    def test_serialize_deserialize(self, conditioning_block):
        block = conditioning_block["make"]()
        block.build(**conditioning_block["build_shapes"])

        serialized = serialize(block)
        deserialized = deserialize(serialized)
        reserialized = serialize(deserialized)

        assert keras.tree.lists_to_tuples(serialized) == keras.tree.lists_to_tuples(reserialized)

    def test_save_and_load(self, tmp_path, conditioning_block):
        block = conditioning_block["make"]()
        block.build(**conditioning_block["build_shapes"])

        keras.saving.save_model(block, tmp_path / "model.keras")
        loaded = keras.saving.load_model(tmp_path / "model.keras")

        assert_layers_equal(block, loaded)

    def test_output_fidelity_after_load(self, tmp_path, conditioning_block):
        block = conditioning_block["make"]()
        block.build(**conditioning_block["build_shapes"])
        x = keras.random.normal(conditioning_block["x_shape"])
        cond = keras.random.normal(conditioning_block["cond_shape"])
        y_before = block((x, cond), training=False)

        keras.saving.save_model(block, tmp_path / "model.keras")
        loaded = keras.saving.load_model(tmp_path / "model.keras")
        y_after = loaded((x, cond), training=False)

        assert keras.ops.all(keras.ops.isclose(y_before, y_after))


class TestConditionalDenseBlock:
    """ConditionalDenseBlock-specific behaviour not shared with the transformer block."""

    @pytest.mark.parametrize("norm", [None, "layer"])
    def test_output_shape_norms(self, norm, cond_build_shapes):
        block = ConditionalDenseBlock(width=8, norm=norm)
        block.build(**cond_build_shapes)
        x = keras.random.normal((4, 6))
        cond = keras.random.normal((4, 3))
        y = block((x, cond), training=False)
        assert y.shape == (4, 8)
