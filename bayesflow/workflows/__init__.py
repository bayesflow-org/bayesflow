r"""
High-level interfaces for amortized Bayesian workflows. :py:class:`~bayesflow.workflows.BasicWorkflow` is a good place
to start; for ensemble-based inference see :py:class:`~bayesflow.workflows.EnsembleWorkflow`.

Examples
--------
>>> import bayesflow as bf
>>> workflow = bf.BasicWorkflow(
...     simulator=bf.simulators.SIR(),
...     inference_network=bf.networks.FlowMatching(),
...     inference_variables=["parameters"],
...     inference_conditions=["observables"],
... )
>>> history = workflow.fit_online(epochs=20, batch_size=32, num_batches_per_epoch=200)  # doctest: +SKIP
>>> diagnostics = workflow.plot_default_diagnostics(test_data=300)  # doctest: +SKIP
"""

from .basic_workflow import BasicWorkflow
from .ensemble_workflow import EnsembleWorkflow
from .compositional_workflow import CompositionalWorkflow

from ..utils._docs import _add_imports_to_all

_add_imports_to_all(include_modules=[])
