import numpy as np

# Mean Squared
def MeanSquared(o, y):
    return np.mean(np.square(o - y))

def MeanSquaredPrime(o, y):
    return o - y

# Cross Entropy
def CrossEntropy(o, y):
    return -np.log(o[range(o.shape[0]), np.argmax(y, axis=1)])

def CrossEntropyPrime(o, y):
    do = np.copy(o)
    do[range(o.shape[0]), np.argmax(y, axis=1)] -= 1
    return do

# Global Loss Dictionary
allLosses = globals()

# Get Loss Function
def Losses(name):
    name = "".join(name.title().split(" "))
    return allLosses[name], allLosses[name + "Prime"]
