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
    do = np.copy(o)
    do[range(o.shape[0]), np.argmax(y, axis=1)] -= 1
    return do

# Global Loss Dictionary
losses = globals()

# Get Loss Function
def get(name):
    return losses[name], losses[name + 'Prime']
