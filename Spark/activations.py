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

# Tanh Activations
def tanh(x):
    return np.tanh(x)

def tanhPrime(x):
    return 1 / (np.cosh(x)**2)

# ReLu Activations
def relu(x):
    return x * (x > 0)

def reluPrime(x):
    y = np.copy(x)
    y[x <= 0] = 0
    return y

# Softmax Activations
def softmax(x):
    exp = np.exp(x - np.max(x))
    return exp / np.sum(exp, axis=1, keepdims=True)

def softmaxPrime(x):
    return x / x.shape[0]

# Global Activations Dictionary
activations = globals()

# Get Activation Function
def get(name):
    if name is None:
        return linear, linearPrime
    else:
        return activations[name], activations[name + 'Prime']
