from .Layer import Layer
from ..activations import activations

import numpy as np

class Activation(Layer):
    def __init__(self, name):
        # Setup Activation Function
        self.activation, self.activationPrime = activations(name)

    def forward(self, X):
        self.o = self.activation(X)
        return self.o

    def backward(self, dO):
        return np.multiply(dO, self.activationPrime(self.o)), []
