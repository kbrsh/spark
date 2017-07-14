import numpy as np
from autodiff import *

def array(*args, **kwargs):
    return np.array(*args, **kwargs)

def zeros(*args, **kwargs):
    return np.zeros(*args, **kwargs)

def ones(*args, **kwargs):
    return np.ones(*args, **kwargs)

def full(*args, **kwargs):
    return np.full(*args, **kwargs)

def add(arr, item):
    return np.add(arr, item)

def multiply(arr, item):
    return np.multiply(arr, item)

def dot(arr, item):
    return np.dot(arr, item)
