import numpy as np

# Linear Activations
def Linear(x):
    return x

def LinearPrime(x):
    return 1

# Sigmoid Activations
def Sigmoid(x):
    return 1 / (1 + np.exp(-x))

def SigmoidPrime(x):
    return np.exp(-x) / ((1 + np.exp(-x)) ** 2)

# Tanh Activations
def Tanh(x):
    return np.tanh(x)

def TanhPrime(x):
    return 1 / (np.cosh(x)**2)

# Softmax Activations
def Softmax(x):
    exp = np.exp(x - np.max(x))
    return exp / np.sum(exp, axis=1, keepdims=True)

def SoftmaxPrime(x):
    return x / x.shape[0]

# Global Activations Dictionary
allActivations = globals()

# Get Activation Function
def Activations(name):
    name = "".join(name.title().split(" "))
    return allActivations[name], allActivations[name + "Prime"]
