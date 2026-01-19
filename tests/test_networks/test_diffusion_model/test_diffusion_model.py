def test_serialize_deserialize_noise_schedule(noise_schedule):
    from bayesflow.utils.serialization import serialize, deserialize

    serialized = serialize(noise_schedule)
    deserialized = deserialize(serialized)
    reserialized = serialize(deserialized)

    assert serialized == reserialized
    t = 0.251
    x = 0.5
    training = True
    assert noise_schedule.get_log_snr(t, training=training) == deserialized.get_log_snr(t, training=training)
    assert noise_schedule.get_t_from_log_snr(t, training=training) == deserialized.get_t_from_log_snr(
        t, training=training
    )
    assert noise_schedule.derivative_log_snr(t, training=False) == deserialized.derivative_log_snr(t, training=False)
    assert noise_schedule.get_drift_diffusion(t, x, training=False) == deserialized.get_drift_diffusion(
        t, x, training=False
    )
    assert noise_schedule.get_alpha_sigma(t) == deserialized.get_alpha_sigma(t)
    assert noise_schedule.get_weights_for_snr(t) == deserialized.get_weights_for_snr(t)


def test_validate_noise_schedule(noise_schedule):
    noise_schedule.validate()


def test_diffusion_guidance():
    from bayesflow.networks import DiffusionModel
    from bayesflow import BasicWorkflow
    from bayesflow.simulators import TwoMoons

    workflow = BasicWorkflow(
        inference_network=DiffusionModel(subnet_kwargs=dict(widths={8, 8})),
        inference_variables=["parameters"],
        inference_conditions=["observables"],
        simulator=TwoMoons(),
    )

    workflow.fit_online(epochs=2, batch_size=3, num_batches_per_epoch=2, verbose=0)
    test_conditions = workflow.simulate(5)
    samples = workflow.sample(num_samples=10, conditions=test_conditions)["parameters"]

    def constraint(z):
        params = workflow.approximator.standardize_layers["inference_variables"](z, forward=False)
        a1 = params[..., 0]
        return a1  # a1 < 0, pick one moon

    samples_guided = workflow.sample(
        num_samples=10, conditions=test_conditions, constraint_guidance=dict(constraints=constraint)
    )["parameters"]
    assert samples_guided.shape == samples.shape
    assert (samples_guided[..., 0] < 0).all()

    def guidance_function(x, time):
        return x * 0  # dummy function

    samples_guided_func = workflow.sample(
        num_samples=10, conditions=test_conditions, guidance_function=guidance_function
    )["parameters"]
    assert samples_guided_func.shape == samples.shape
