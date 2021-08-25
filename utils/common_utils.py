from IPython import get_ipython
from torch import Tensor
import numpy as np


def get_numpy_from_torch(t: Tensor) -> np.array:
    return t.cpu().numpy()


def in_ipynb():
    try:
        cfg = get_ipython().config
        if cfg['IPKernelApp']['parent_appname'] == 'ipython-notebook':
            return True
        else:
            return False
    except (NameError, AttributeError):
        return False
