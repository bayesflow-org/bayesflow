from collections.abc import Callable
from typing import Any

import numpy as np


LogProb = np.ndarray | Callable[[Any], np.ndarray]
