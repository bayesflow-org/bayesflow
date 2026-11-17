import keras
import io
from contextlib import redirect_stdout

from tests.utils import assert_models_equal


def test_build(estimator, train_dataset):
    assert estimator.built is False

    data_shapes = keras.tree.map_structure(keras.ops.shape, train_dataset[0])
    estimator.build(data_shapes)

    assert estimator.built is True
    assert estimator.inference_network.built is True
    if estimator.summary_network is not None:
        assert estimator.summary_network.built is True


def test_build_from_data(estimator, train_dataset):
    assert estimator.built is False

    estimator.build_from_data(train_dataset[0])

    assert estimator.built is True
    assert estimator.inference_network.built is True
    if estimator.summary_network is not None:
        assert estimator.summary_network.built is True


def test_build_adapter():
    from bayesflow.estimators import RatioEstimator

    _ = RatioEstimator.build_adapter(
        inference_conditions=["inference_conditions"],
        inference_variables=["inference_variables"],
    )


def test_contrastive_sampling(adapter, simulator, estimator):
    joint_samples = simulator.sample(8)
    joint_samples = adapter(joint_samples)
    iv = joint_samples["inference_variables"]
    civ = estimator._sample_from_batch(iv)

    shape = list(keras.ops.shape(iv))
    shape.insert(1, estimator.K)
    assert list(keras.ops.shape(civ)) == shape

    for k in range(estimator.K):
        assert not keras.ops.all(keras.ops.isclose(iv, civ[:, k]))


def test_compute_metrics_with_fewer_samples_than_k(adapter, simulator, estimator):
    batch = adapter(simulator.sample(estimator.K - 1))
    estimator.build_from_data(batch)

    metrics = estimator.compute_metrics(**batch)

    assert keras.ops.isfinite(metrics["loss"])


def test_fit(estimator, train_dataset, validation_dataset):
    estimator.compile(optimizer="AdamW")
    num_epochs = 1

    # Capture ostream and train model
    with io.StringIO() as stream:
        with redirect_stdout(stream):
            estimator.fit(dataset=train_dataset, validation_data=validation_dataset, epochs=num_epochs)

        output = stream.getvalue()

    # check that the loss is shown
    assert "loss" in output

    # TODO: reliably check that the log ratio of a joint sample is larger than that of a marginal sample
    # batch = next(iter(train_dataset))
    #
    # print(type(batch))
    # print(list(batch.keys()))
    # print(keras.tree.map_structure(keras.ops.shape, batch))
    #
    # joint_iv = batch["inference_variables"]
    # joint_ic = estimator.summary_network(batch["inference_conditions"])
    # marginal_iv = estimator._sample_from_batch(joint_iv, seed=random_seed)[:, 0]
    # marginal_ic = joint_ic
    #
    # joint_ratio = estimator.logits(joint_iv, joint_ic, stage="inference")
    # marginal_ratio = estimator.logits(marginal_iv, marginal_ic, stage="inference")
    #
    # assert keras.ops.mean(joint_ratio) > keras.ops.mean(marginal_ratio)


def test_save_and_load(tmp_path, estimator, train_dataset):
    # to save, the model must be built
    data_shapes = keras.tree.map_structure(keras.ops.shape, train_dataset[0])
    estimator.build(data_shapes)
    estimator.compute_metrics(**train_dataset[0])

    keras.saving.save_model(estimator, tmp_path / "model.keras")
    loaded = keras.saving.load_model(tmp_path / "model.keras")

    assert_models_equal(estimator, loaded)
