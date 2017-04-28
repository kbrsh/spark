import numpy as np

# Linear Activations
def linear(x):
    return x

def linearPrime(x):
    return 1

# Sigmoid Activations
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoidPrime(x):
    return np.exp(-x) / ((1 + np.exp(-x)) ** 2)

# Sigmoid Activations
def tanh(x):
    return np.tanh(x)

def tanhPrime(x):
    return 1 / (np.cosh(x)**2)

# Global Activations Dictionary
activations = globals()

# Get Activation Function
def get(name):
    if name is None:
        return linear, linearPrime
    else:
        return activations[name], activations[name + 'Prime']
