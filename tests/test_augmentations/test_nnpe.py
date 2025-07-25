def test_nnpe(random_data):
    import numpy as np
    from bayesflow.augmentations import NNPE

    # Test basic case with global noise application
    aug = NNPE(spike_scale=1.0, slab_scale=1.0, per_dimension=False, seed=42)
    result = aug(random_data["x1"])

    # Check that result is the same shape as x1, but changed
    assert result.shape == random_data["x1"].shape
    assert not np.allclose(result, random_data["x1"])

    # Test both scales and seed are None case (automatic scale determination) with dimensionwise noise application
    aug_auto = NNPE(slab_scale=None, spike_scale=None, per_dimension=True, seed=None)
    result_auto = aug_auto(random_data["x2"])
    assert result_auto.shape == random_data["x2"].shape
    assert not np.allclose(result_auto, random_data["x2"])

    # Test dimensionwise versus global noise application (per_dimension=True vs per_dimension=False)
    # Create data with second dimension having higher variance
    data_shape = (32, 16, 1)
    rng = np.random.default_rng(42)
    zero = np.ones(shape=data_shape)
    high = rng.normal(0, 100.0, size=data_shape)
    var_data = np.concatenate([zero, high], axis=-1)

    # Apply dimensionwise and global adapters with automatic slab_scale scale determination
    aug_partial_global = NNPE(spike_scale=0, slab_scale=None, per_dimension=False, seed=42)
    aug_partial_dim = NNPE(spike_scale=[0, 1], slab_scale=None, per_dimension=True, seed=42)
    result_dim = aug_partial_dim(var_data)
    result_glob = aug_partial_global(var_data)

    # Compute standard deviations of noise per last axis dimension
    noise_dim = result_dim - var_data
    noise_glob = result_glob - var_data
    std_dim = np.std(noise_dim, axis=(0, 1))
    std_glob = np.std(noise_glob, axis=(0, 1))

    # Dimensionwise should assign zero noise, global some noise to zero-variance dimension
    assert std_dim[0] == 0
    assert std_glob[0] > 0
    # Both should assign noise to high-variance dimension
    assert std_dim[1] > 0
    assert std_glob[1] > 0
