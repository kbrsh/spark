from __future__ import absolute_import

from .. import activations

import numpy as np

class Activation(object):
    def __init__(self, name):
        # Setup Activation Function
        self.activation, self.activationPrime = activations.get(name)

    def loss(self, o, y):
        return np.mean(np.square(o - y), axis=-1)

    def lossPrime(self, o, y):
        return o - y

    def forward(self, X):
        o = self.activation(X)
        self.o = o
        return o

    def backward(self, dY):
        return self.activationPrime(self.o)
