import numpy as np

from ..multilevel_simulator import MultilevelSimulator


def prior():
    beta = np.random.normal([2, 0], [3, 1])
    sigma = np.random.gamma(1, 1)

    return {"beta": beta, "sigma": sigma}


def likelihood(beta, sigma):
    x = np.random.normal(0, 1)
    y = np.random.normal(beta[0] + beta[1] * x, sigma)

    return {"x": x, "y": y}


def meta():
    N = np.random.randint(5, 15)

    return {"N": N}


def single_level_simulator():
    """
    Simple single-level regression simulator. For a more general example, see
    https://bayesflow.org/main/_examples/Bayesian_GLM.html

    The sample functions draw one sample per call.
    """
    simulator = MultilevelSimulator(meta_fn=meta)

    simulator.add_node("prior", sample_fn=prior)
    simulator.add_node("likelihood", sample_fn=likelihood, reps="N")

    simulator.add_edge("prior", "likelihood")

    return simulator
