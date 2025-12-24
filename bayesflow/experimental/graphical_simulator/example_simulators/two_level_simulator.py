import numpy as np

from ..graphical_simulator import GraphicalSimulator


def two_level_simulator(repeated_roots=False):
    r"""
    Simple hierarchical model with two levels of parameters: hyperparameters
    and local parameters, along with a shared parameter:

    hypers
       |
    locals  shared
        \    /
         \  /
          y

    Parameters
    ----------
    repeated_roots : bool, default false.

    """

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

    simulator = GraphicalSimulator()

    if not repeated_roots:
        simulator.add_node("hypers", sample_fn=sample_hypers)
    else:
        simulator.add_node("hypers", sample_fn=sample_hypers, reps=5)

    simulator.add_node(
        "locals",
        sample_fn=sample_locals,
        reps=6,
    )

    simulator.add_node("shared", sample_fn=sample_shared)
    simulator.add_node("y", sample_fn=sample_y, reps=10)

    simulator.add_edge("hypers", "locals")
    simulator.add_edge("locals", "y")
    simulator.add_edge("shared", "y")

    return simulator
