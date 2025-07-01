import numpy as np

from ..graphical_simulator import GraphicalSimulator


def single_level_simulator():
    """
    Simple single-level simulator that implements the same model as in
    https://bayesflow.org/main/_examples/Linear_Regression_Starter.html
    """

    def prior():
        beta = np.random.normal([2, 0], [3, 1])
        sigma = np.random.gamma(1, 1)

        return {"beta": beta, "sigma": sigma}

    def likelihood(beta, sigma, N):
        x = np.random.normal(0, 1, size=N)
        y = np.random.normal(beta[0] + beta[1] * x, sigma, size=N)

        return {"x": x, "y": y}

    def meta():
        N = np.random.randint(5, 15)

        return {"N": N}

    simulator = GraphicalSimulator(meta_fn=meta)

    simulator.add_node("prior", sample_fn=prior)
    simulator.add_node("likelihood", sample_fn=likelihood)

    simulator.add_edge("prior", "likelihood")

    return simulator
