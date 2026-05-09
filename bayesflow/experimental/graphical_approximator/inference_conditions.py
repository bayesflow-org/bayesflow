from bayesflow.types import Tensor

from .tensor_concatenation import concatenate


def gather_node_output(
    variable_names: list[str],
    simulation_output: dict[str, Tensor],
) -> Tensor:
    """
    Collects all output tensors for a node from `simulation_output` and
    concatenates them along the last axis.

    Parameters
    ----------
    variable_names : list[str]
        The variable names produced by the node, typically one entry of
        ``SimulationGraph.variable_names()``.
    simulation_output : dict[str, Tensor]
        Dictionary mapping variable names to tensors, as produced by the simulator.

    Returns
    -------
    Tensor
        Concatenation of all tensors along the last axis.
    """
    tensors = [simulation_output[name] for name in variable_names]
    return concatenate(tensors)
