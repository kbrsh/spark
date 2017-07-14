import util as sp

# Linear Activations
def linear(x):
    return x

def linearPrime(x):
    return 1

# Sigmoid Activations
def sigmoid(x):
    return 1 / (1 + sp.exp(-x))

def sigmoidPrime(x):
    return sp.exp(-x) / ((1 + sp.exp(-x)) ** 2)

# Tanh Activations
def tanh(x):
    return sp.tanh(x)

def tanhPrime(x):
    return 1 / (sp.cosh(x)**2)

# Softmax Activations
def softmax(x):
    exp = sp.exp(x - sp.max(x))
    return exp / sp.sum(exp, axis=1, keepdims=True)

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
