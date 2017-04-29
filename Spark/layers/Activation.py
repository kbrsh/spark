from . import Layer
from .. import activations

import numpy as np

class Activation(Layer.Layer):
    def __init__(self, name):
        # Setup Activation Function
        self.activation, self.activationPrime = activations.get(name)

    def forward(self, X):
        self.o = self.activation(X)
        return self.o

    def backward(self, dO):
        return np.multiply(dO, self.activationPrime(self.o)), []
