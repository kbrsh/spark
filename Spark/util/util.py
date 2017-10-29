import numpy as np
from .diff import variable, gradient

def array(*args):
    return np.array(*args)

def random(*args):
    return np.random.randn(*args)

def zeros(*args):
    return np.zeros(*args)

def ones(*args):
    return np.ones(*args)

def add(arr, item):
    result = arr.add(item)
    return result

def subtract(arr, item):
    result = arr.subtract(item)
    return result

def multiply(arr, item):
    result = arr.multiply(item)
    return result

def dot(arr, item):
    result = arr.dot(item)
    return result
