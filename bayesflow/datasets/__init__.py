r"""
A collection of `keras.utils.PyDataset <https://keras.io/api/utils/python_utils/#pydataset-class>`__\ s, which
wrap your data-generating process (i.e., your :py:class:`~bayesflow.simulators.Simulator`) and thus determine the
effective training strategy (e.g., online or offline).
"""

from .online_dataset import OnlineDataset
from .offline_dataset import OfflineDataset
from .disk_dataset import DiskDataset

from .ensemble_dataset import EnsembleDataset
from .ensemble_online_dataset import EnsembleOnlineDataset
from .ensemble_indexed_dataset import EnsembleIndexedDataset


from ..utils._docs import _add_imports_to_all

_add_imports_to_all(include_modules=[])
