import keras
import pytest
from tests.utils import assert_models_equal


def test_save_and_load(tmp_path, approximator, train_dataset, validation_dataset):
    # to save, the model must be built
    data_shapes = keras.tree.map_structure(keras.ops.shape, train_dataset[0])
    approximator.build(data_shapes)
    for layer in approximator.standardize_layers.values():
        assert layer.built
        for count in layer.count:
            assert count == 0.0
    approximator.compute_metrics(**train_dataset[0])

    keras.saving.save_model(approximator, tmp_path / "model.keras")
    loaded = keras.saving.load_model(tmp_path / "model.keras")

    assert_models_equal(approximator, loaded)


def test_save_and_load_all_variants(
    tmp_path, adapter, inference_network, summary_network, train_dataset, validation_dataset
):
    """Run the same save/load assertions for all `standardize` options in one test node.

    This avoids relying on pytest's bracketed node-id selection; it constructs
    an approximator for each `standardize` value and runs the same checks.
    Any failures across variants are aggregated and reported at the end.
    """
    from bayesflow import ContinuousApproximator

    standardize_values = [
        "all",
        None,
        "inference_variables",
        "summary_variables",
        ("inference_variables", "summary_variables", "inference_conditions"),
    ]

    failures = []

    for standardize in standardize_values:
        approximator = ContinuousApproximator(
            adapter=adapter,
            inference_network=inference_network,
            summary_network=summary_network,
            standardize=standardize,
        )

        try:
            data_shapes = keras.tree.map_structure(keras.ops.shape, train_dataset[0])
            approximator.build(data_shapes)
            for layer in approximator.standardize_layers.values():
                assert layer.built
                for count in layer.count:
                    assert count == 0.0
            approximator.compute_metrics(**train_dataset[0])

            model_path = tmp_path / f"model_{str(standardize)}.keras"
            keras.saving.save_model(approximator, model_path)
            loaded = keras.saving.load_model(model_path)

            assert_models_equal(approximator, loaded)
        except Exception as exc:  # collect failures and continue
            failures.append((standardize, repr(exc)))

    if failures:
        msgs = ", ".join([f"{s!r}: {m}" for s, m in failures])
        pytest.fail(f"One or more standardize variants failed: {msgs}")
