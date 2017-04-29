import numpy as np

# Mean Squared
def meanSquared(o, y):
    return np.mean(np.square(o - y))

def meanSquaredPrime(o, y):
    return o - y

# Global Loss Dictionary
losses = globals()

# Get Loss Function
def get(name):
    return losses[name], losses[name + 'Prime']
