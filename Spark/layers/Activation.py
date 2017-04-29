from . import Layer
from .. import activations

import numpy as np

class Activation(Layer.Layer):
    def __init__(self, name):
        # Setup Activation Function
        self.activation, self.activationPrime = activations.get(name)

    def forward(self, X):
        o = self.activation(X)
        self.o = o
        return o

    def backward(self, dY):
        return np.multiply(dY, self.activationPrime(self.o)), []
