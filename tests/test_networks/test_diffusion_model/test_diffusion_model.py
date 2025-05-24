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
