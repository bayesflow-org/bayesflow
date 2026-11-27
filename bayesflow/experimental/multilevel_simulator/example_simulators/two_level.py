import math

import keras
import numpy as np

from ..multilevel_simulator import MultilevelSimulator


def sample_hypers():
    hyper_mean = np.random.normal()
    hyper_std = np.abs(np.random.normal())

    return {"hyper_mean": hyper_mean, "hyper_std": hyper_std}


def sample_locals(hyper_mean, hyper_std):
    local_mean = np.random.normal(hyper_mean, hyper_std)

    return {"local_mean": local_mean}


def sample_shared():
    shared_std = np.abs(np.random.normal())

    return {"shared_std": shared_std}


def sample_y(local_mean, shared_std):
    y = np.random.normal(local_mean, shared_std)

    return {"y": y}


def normal_log_prob(x, mean, std):
    z = (x - mean) / std

    return keras.ops.sum(-0.5 * z**2 - keras.ops.log(std) - 0.5 * math.log(2 * math.pi), axis=-1)


def hypers_log_prob(hyper_mean, hyper_std):
    return normal_log_prob(hyper_mean, 0.0, 1.0) + math.log(2) + normal_log_prob(hyper_std, 0.0, 1.0)


def locals_log_prob(local_mean, hyper_mean, hyper_std):
    return normal_log_prob(local_mean, hyper_mean, hyper_std)


def shared_log_prob(shared_std):
    return math.log(2) + normal_log_prob(shared_std, 0.0, 1.0)


def y_log_prob(y, local_mean, shared_std):
    return normal_log_prob(y, local_mean, shared_std)


def two_level_simulator():
    r"""
    Simple hierarchical model with two levels of parameters: hyperparameters
    and local parameters, along with a shared parameter:

    hypers
       |
    locals  shared
        \    /
         \  /
          y

    The sample functions draw one sample per call.
    """
    simulator = MultilevelSimulator()

    simulator.add_node("hypers", sample_fn=sample_hypers, log_prob_fn=hypers_log_prob)
    simulator.add_node("locals", sample_fn=sample_locals, log_prob_fn=locals_log_prob, reps=6)
    simulator.add_node("shared", sample_fn=sample_shared, log_prob_fn=shared_log_prob)
    simulator.add_node("y", sample_fn=sample_y, log_prob_fn=y_log_prob, reps=10)

    simulator.add_edge("hypers", "locals")
    simulator.add_edge("locals", "y")
    simulator.add_edge("shared", "y")

    return simulator
