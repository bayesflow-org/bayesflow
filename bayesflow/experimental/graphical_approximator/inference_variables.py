from typing import TYPE_CHECKING

from bayesflow.types import Tensor

from .tensor_concatenation import concatenate

if TYPE_CHECKING:
    from .graphical_approximator import GraphicalApproximator


def inference_variables_by_network(approximator: "GraphicalApproximator", adapted_data: dict) -> dict[int, Tensor]:
    """
    Collects and concatenates the inference variables for each network.

    For each network, gathers the simulated parameter tensors assigned to that
    network (via ``GraphicalApproximator.network_composition``), optionally
    standardizes them, and concatenates them into a single tensor along the
    last axis.

    Parameters
    ----------
    approximator : GraphicalApproximator
        The approximator, providing network composition, variable names, and
        standardization layers.
    adapted_data : dict[str, Tensor]
        Dictionary mapping variable names to tensors, as produced by the
        adapter.

    Returns
    -------
    dict[int, Tensor]
        Mapping from network index to the concatenated inference variable
        tensor for that network. Networks whose variables are not present in
        ``adapted_data`` are omitted.
    """
    result = {}

    for network_idx in range(len(approximator.inference_networks)):
        try:
            result[network_idx] = _collect_inference_variables(approximator, adapted_data, network_idx)
        except KeyError:
            pass

    return result


def _collect_inference_variables(
    approximator: "GraphicalApproximator",
    adapted_data: dict,
    network_idx: int,
) -> Tensor:
    """
    Returns the concatenated inference variable tensor for ``network_idx``.

    Raises ``KeyError`` if any required variable is missing from
    ``adapted_data``.
    """
    tensors = []

    for node in approximator.network_composition[network_idx]:
        for name in approximator.variable_names[node]:
            var = adapted_data[name]

            if name in approximator.standardize:
                var = approximator.standardize_layers[name](var, stage="validation")

            tensors.append(var)

    return concatenate(tensors)
