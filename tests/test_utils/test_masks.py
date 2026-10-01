import keras
import numpy as np

from bayesflow.networks import MLP
from bayesflow.utils.masks import sample_input_masks


def _sample(fixed_target_mask=None):
    x = keras.random.normal((4, 3))
    return sample_input_masks(
        MLP(widths=(8,)),
        x,
        None,
        {},
        True,
        fixed_target_prob=0.5,
        missing_target_prob=0.0,
        missing_conditions_prob=0.0,
        seed_generator=keras.random.SeedGenerator(0),
        fixed_target_mask=fixed_target_mask,
    )


def test_given_fixed_target_mask_replaces_random_draw():
    mask = np.array([[1, 1, 0]] * 4, dtype="float32")
    mask_x, loss_mask, _ = _sample(mask)

    np.testing.assert_array_equal(keras.ops.convert_to_numpy(mask_x), mask)
    np.testing.assert_array_equal(keras.ops.convert_to_numpy(loss_mask), mask)


def test_unbatched_fixed_target_mask_is_broadcast():
    mask_x, _, _ = _sample(np.array([1, 0, 1], dtype="float32"))

    assert keras.ops.shape(mask_x) == (4, 3)
    np.testing.assert_array_equal(keras.ops.convert_to_numpy(mask_x), np.array([[1, 0, 1]] * 4, dtype="float32"))


def test_random_draw_without_fixed_target_mask():
    mask_x, loss_mask, _ = _sample()

    assert keras.ops.shape(mask_x) == (4, 3)
    assert mask_x is loss_mask
