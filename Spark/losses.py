import numpy as np

# Mean Squared
def meanSquared(o, y):
    return np.mean(np.square(o - y))

def meanSquaredPrime(o, y):
    return o - y

# Cross Entropy
def crossEntropy(o, y):
    return -np.log(o[range(o.shape[0]), np.argmax(y, axis=1)])

def crossEntropyPrime(o, y):
    return (o - y) / o.shape[0]

# Global Loss Dictionary
losses = globals()

# Get Loss Function
def get(name):
    return losses[name], losses[name + 'Prime']
