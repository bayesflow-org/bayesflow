"""HNN model, simulation, and data helpers for the integration tutorial."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
from scipy.stats import gamma, lognorm, qmc


os.environ.setdefault("NEURON_MODULE_OPTIONS", "-nogui")

HNN_CORE_VERSION = "0.6.1"
TIME_GRID = np.arange(10.0, 169.0, 2.0)
TSTOP = 190.0
NOISE_SD = 0.5

PARAMETER_NAMES = [
    "early_feedforward_time",
    "feedback_delay",
    "late_feedforward_delay",
    "early_feedforward_gain",
    "feedback_gain",
    "late_feedforward_gain",
    "observation_gain",
]
TIMING_NAMES = PARAMETER_NAMES[:3]
GAIN_NAMES = PARAMETER_NAMES[3:6]
PARAMETER_LABELS = [
    "Early feedforward time (ms)",
    "Feedback delay (ms)",
    "Late feedforward delay (ms)",
    "Early feedforward gain",
    "Feedback gain",
    "Late feedforward gain",
    "Dipole amplitude scale",
]
PRIOR_SPECS = {
    "early_feedforward_time": {"family": "lognormal", "median": 26.61, "log_sd": 0.17},
    "feedback_delay": {"family": "lognormal", "median": 63.53 - 26.61, "log_sd": 0.25},
    "late_feedforward_delay": {"family": "lognormal", "median": 137.12 - 63.53, "log_sd": 0.17},
    "early_feedforward_gain": {"family": "lognormal", "median": 1.0, "log_sd": 0.70},
    "feedback_gain": {"family": "lognormal", "median": 1.0, "log_sd": 0.70},
    "late_feedforward_gain": {"family": "lognormal", "median": 1.0, "log_sd": 0.70},
    "observation_gain": {"family": "gamma", "shape": 2.0, "scale": 2.0},
}
REFERENCE_PARAMETERS = {
    "early_feedforward_time": 26.61,
    "feedback_delay": 63.53 - 26.61,
    "late_feedforward_delay": 137.12 - 63.53,
    "early_feedforward_gain": 1.0,
    "feedback_gain": 1.0,
    "late_feedforward_gain": 1.0,
    "observation_gain": 1.0,
}

REFERENCE_WEIGHTS = {
    "early": {
        "ampa": {
            "L2_basket": 0.08831,
            "L2_pyramidal": 0.01525,
            "L5_basket": 0.19934,
            "L5_pyramidal": 0.00865,
        },
        "nmda": None,
    },
    "feedback": {
        "ampa": {
            "L2_basket": 0.006562,
            "L2_pyramidal": 0.000007,
            "L5_pyramidal": 0.142300,
        },
        "nmda": {
            "L2_basket": 0.019482,
            "L2_pyramidal": 0.004317,
            "L5_pyramidal": 0.080074,
        },
    },
    "late": {
        "ampa": {
            "L2_basket": 0.000003,
            "L2_pyramidal": 1.438840,
            "L5_basket": 0.008958,
            "L5_pyramidal": 0.684013,
        },
        "nmda": None,
    },
}
PROXIMAL_DELAYS = {
    "L2_basket": 0.1,
    "L2_pyramidal": 0.1,
    "L5_basket": 1.0,
    "L5_pyramidal": 1.0,
}
DISTAL_DELAYS = {
    "L2_basket": 0.1,
    "L2_pyramidal": 0.1,
    "L5_pyramidal": 0.1,
}

DATA_PATH = Path(__file__).with_name("hnn_integration_fast.npz")
EXPERIMENTAL_DATA_PATH = Path(__file__).with_name("yes_trial_S1_ERP_all_avg.txt")

_SPLIT_SIZES = {"train": 4096, "validation": 512, "test": 256}
_EXPECTED_CACHE_METADATA = {
    "hnn_core_version": HNN_CORE_VERSION,
    "model": "neymotin_2020_model",
    "mesh_shape": [3, 3],
    "tstop_ms": TSTOP,
    "dt_ms": 0.025,
    "smooth_ms": 20.0,
    "dipole_scale": 3000.0,
    "measurement_noise_sd_nAm": NOISE_SD,
    "design": "LatinHypercube",
    "prior_specs": PRIOR_SPECS,
    "parameter_names": PARAMETER_NAMES,
    "split_sizes": _SPLIT_SIZES,
}


def _scaled(profile, gain):
    if profile is None:
        return None
    return {cell_type: gain * weight for cell_type, weight in profile.items()}


def _set_hnn_parameters(parameters, net):
    early_time = float(parameters["early_feedforward_time"])
    feedback_time = early_time + float(parameters["feedback_delay"])
    late_time = feedback_time + float(parameters["late_feedforward_delay"])
    early_gain = float(parameters["early_feedforward_gain"])
    feedback_gain = float(parameters["feedback_gain"])
    late_gain = float(parameters["late_feedforward_gain"])
    event_seed = int(np.random.randint(1, np.iinfo(np.int32).max - 2))

    net.add_evoked_drive(
        "evprox1",
        mu=early_time,
        sigma=2.47,
        numspikes=1,
        location="proximal",
        n_drive_cells="n_cells",
        cell_specific=True,
        weights_ampa=_scaled(REFERENCE_WEIGHTS["early"]["ampa"], early_gain),
        synaptic_delays=PROXIMAL_DELAYS,
        event_seed=event_seed,
    )
    net.add_evoked_drive(
        "evdist1",
        mu=feedback_time,
        sigma=3.85,
        numspikes=1,
        location="distal",
        n_drive_cells="n_cells",
        cell_specific=True,
        weights_ampa=_scaled(REFERENCE_WEIGHTS["feedback"]["ampa"], feedback_gain),
        weights_nmda=_scaled(REFERENCE_WEIGHTS["feedback"]["nmda"], feedback_gain),
        synaptic_delays=DISTAL_DELAYS,
        event_seed=event_seed + 1,
    )
    net.add_evoked_drive(
        "evprox2",
        mu=late_time,
        sigma=8.33,
        numspikes=1,
        location="proximal",
        n_drive_cells="n_cells",
        cell_specific=True,
        weights_ampa=_scaled(REFERENCE_WEIGHTS["late"]["ampa"], late_gain),
        synaptic_delays=PROXIMAL_DELAYS,
        event_seed=event_seed + 2,
    )


def _waveform_from_result(result):
    observation_gain = float(result["param_values"]["observation_gain"])
    dipole = result["dpl"][0].copy().smooth(window_len=20).scale(3000 * observation_gain)
    waveform = np.interp(TIME_GRID, dipole.times, dipole.data["agg"])
    waveform -= waveform[TIME_GRID <= 20.0].mean()
    return waveform + np.random.normal(0.0, NOISE_SD, TIME_GRID.size)


def _summarize_batch(results):
    return [{"dipole": _waveform_from_result(result)} for result in results]


def simulate_parameter_batch(parameter_draws, *, n_jobs=4):
    """Simulate paired parameter rows and return aggregate dipole waveforms."""
    from hnn_core import neymotin_2020_model
    from hnn_core.batch_simulate import BatchSimulate

    parameter_grid = {name: parameter_draws[name].ravel() for name in PARAMETER_NAMES}
    batch = BatchSimulate(
        set_params=_set_hnn_parameters,
        net=neymotin_2020_model(mesh_shape=(3, 3)),
        tstop=TSTOP,
        dt=0.025,
        n_trials=1,
        batch_size=64,
        clear_cache=True,
        summary_func=_summarize_batch,
    )
    output = batch.run(parameter_grid, combinations=False, n_jobs=n_jobs, backend="loky", verbose=False)
    summaries = [item for batch_summaries in output["summary_statistics"] for item in batch_summaries]
    return np.stack([item["dipole"] for item in summaries], dtype=np.float32)


def _draw_prior(n_simulations):
    unit_cube = qmc.LatinHypercube(d=len(PARAMETER_NAMES)).random(n_simulations)
    draws = {}

    for column, name in enumerate(PARAMETER_NAMES):
        spec = PRIOR_SPECS[name]
        if spec["family"] == "lognormal":
            values = lognorm.ppf(unit_cube[:, column], s=spec["log_sd"], scale=spec["median"])
        else:
            values = gamma.ppf(unit_cube[:, column], a=spec["shape"], scale=spec["scale"])
        draws[name] = values[:, None]

    return draws


def _generate_split(n_simulations, n_jobs):
    parameters = _draw_prior(n_simulations)
    dipoles = simulate_parameter_batch(parameters, n_jobs=n_jobs)
    return {
        **{name: np.array(values, dtype=np.float32) for name, values in parameters.items()},
        "dipole": dipoles,
    }


def _write_cache(n_jobs):
    generated = {}
    for name, size in _SPLIT_SIZES.items():
        print(f"Simulating {name}: {size:,}")
        generated[name] = _generate_split(size, n_jobs)

    payload = {
        "time_grid": np.array(TIME_GRID, dtype=np.float32),
        "metadata": np.array(json.dumps(_EXPECTED_CACHE_METADATA, sort_keys=True)),
    }

    for split, split_data in generated.items():
        for name, values in split_data.items():
            payload[f"{split}_{name}"] = values

    np.savez_compressed(DATA_PATH, **payload)


def _read_cache():
    if not DATA_PATH.is_file():
        raise FileNotFoundError(f"Missing {DATA_PATH}. Call load_dataset(regenerate=True) to build it.")

    with np.load(DATA_PATH, allow_pickle=False) as archive:
        if not np.allclose(archive["time_grid"], TIME_GRID):
            raise ValueError("The cached waveform grid does not match TIME_GRID.")

        metadata = json.loads(archive["metadata"].item())
        splits = {
            split: {name: archive[f"{split}_{name}"].copy() for name in [*PARAMETER_NAMES, "dipole"]}
            for split in _SPLIT_SIZES
        }

    for key, expected in _EXPECTED_CACHE_METADATA.items():
        if metadata.get(key) != expected:
            raise ValueError(f"Cached {key}={metadata.get(key)!r}, expected {expected!r}.")

    for split_name, split in splits.items():
        expected_rows = _SPLIT_SIZES[split_name]
        expected_shapes = {
            **{name: (expected_rows, 1) for name in PARAMETER_NAMES},
            "dipole": (expected_rows, TIME_GRID.size),
        }
        for name, expected_shape in expected_shapes.items():
            if split[name].shape != expected_shape or not np.isfinite(split[name]).all():
                raise ValueError(f"Invalid {split_name}_{name}: expected finite {expected_shape}.")
            if name in PARAMETER_NAMES and not np.all(split[name] > 0):
                raise ValueError(f"Invalid {split_name}_{name}: expected strictly positive values.")

    return splits["train"], splits["validation"], splits["test"], metadata


def load_dataset(*, regenerate=False, n_jobs=4):
    """Load the checked-in simulation cache, optionally replacing it first."""
    if regenerate:
        _write_cache(n_jobs)
    return _read_cache()


def load_experimental_waveform():
    """Load, interpolate, and center the recorded tactile-MEG waveform."""
    recorded = np.loadtxt(EXPERIMENTAL_DATA_PATH)
    waveform = np.interp(TIME_GRID, recorded[:, 0], recorded[:, 1])
    waveform -= waveform[TIME_GRID <= 20.0].mean()
    return np.array(waveform, dtype=np.float32)


__all__ = [
    "DATA_PATH",
    "GAIN_NAMES",
    "HNN_CORE_VERSION",
    "NOISE_SD",
    "PARAMETER_LABELS",
    "PARAMETER_NAMES",
    "PRIOR_SPECS",
    "REFERENCE_PARAMETERS",
    "TIME_GRID",
    "TIMING_NAMES",
    "load_dataset",
    "load_experimental_waveform",
    "simulate_parameter_batch",
]
