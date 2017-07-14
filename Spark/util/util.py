import numpy as np
from autodiff import *

def array(*args, **kwargs):
    return np.array(*args, **kwargs)

def zeros(*args, **kwargs):
    return np.zeros(*args, **kwargs)

def ones(*args, **kwargs):
    return np.ones(*args, **kwargs)

def add(arr, item):
    return np.add(arr, item)
