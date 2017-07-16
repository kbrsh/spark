import numpy as np
from autodiff import *

def add(arr, item):
    result = arr.__add__(item)
    return result

def subtract(arr, item):
    result = arr.__sub__(item)
    return result

def multiply(arr, item):
    result = arr.__mul__(item)
    return result

def dot(arr, item):
    result = arr.__dot__(item)
    return result
