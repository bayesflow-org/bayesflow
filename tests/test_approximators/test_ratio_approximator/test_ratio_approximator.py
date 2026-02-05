import keras
import io
from contextlib import redirect_stdout

from tests.utils import assert_models_equal


def test_build(approximator, train_dataset):
    assert approximator.built is False

    data_shapes = keras.tree.map_structure(keras.ops.shape, train_dataset[0])
    approximator.build(data_shapes)

    assert approximator.built is True
    assert approximator.classifier_network.built is True
    if approximator.summary_network is not None:
        assert approximator.summary_network.built is True


def test_build_from_data(approximator, train_dataset):
    assert approximator.built is False

    approximator.build_from_data(train_dataset[0])

    assert approximator.built is True
    assert approximator.classifier_network.built is True
    if approximator.summary_network is not None:
        assert approximator.summary_network.built is True


def test_build_adapter():
    from bayesflow.approximators import RatioApproximator

    _ = RatioApproximator.build_adapter(
        inference_conditions=["inference_conditions"],
        inference_variables=["inference_variables"],
    )


def test_contrastive_sampling(random_seed, adapter, simulator, approximator):
    joint_samples = simulator.sample(32)
    joint_samples = adapter(joint_samples)
    iv = joint_samples["inference_variables"]
    civ = approximator._sample_from_batch(iv, seed=random_seed)

    shape = list(keras.ops.shape(iv))
    shape.insert(1, approximator.K)
    assert list(keras.ops.shape(civ)) == shape

    for k in range(approximator.K):
        assert not keras.ops.all(keras.ops.isclose(iv, civ[:, k]))


def test_fit(random_seed, approximator, train_dataset, validation_dataset, simulator, adapter):
    approximator.compile(optimizer="AdamW")
    num_epochs = 1

    # Capture ostream and train model
    with io.StringIO() as stream:
        with redirect_stdout(stream):
            approximator.fit(dataset=train_dataset, validation_data=validation_dataset, epochs=num_epochs)

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
    # joint_ic = approximator.summary_network(batch["inference_conditions"])
    # marginal_iv = approximator._sample_from_batch(joint_iv, seed=random_seed)[:, 0]
    # marginal_ic = joint_ic
    #
    # joint_ratio = approximator.logits(joint_iv, joint_ic, stage="inference")
    # marginal_ratio = approximator.logits(marginal_iv, marginal_ic, stage="inference")
    #
    # assert keras.ops.mean(joint_ratio) > keras.ops.mean(marginal_ratio)


def test_save_and_load(tmp_path, approximator, train_dataset):
    # to save, the model must be built
    data_shapes = keras.tree.map_structure(keras.ops.shape, train_dataset[0])
    approximator.build(data_shapes)
    approximator.compute_metrics(**train_dataset[0])

    keras.saving.save_model(approximator, tmp_path / "model.keras")
    loaded = keras.saving.load_model(tmp_path / "model.keras")

    assert_models_equal(approximator, loaded)
