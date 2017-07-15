import numpy as np
from autodiff import *

def add(arr, item):
    return np.add(arr, item)

def multiply(arr, item):
    result = arr.__mul__(item)
    return result

def dot(arr, item):
    return np.dot(arr, item)
