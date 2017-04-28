from __future__ import absolute_import

from .. import activations

import numpy as np

class Dense(object):
    def __init__(self, inputSize, outputSize, activation=None, learningRate=1e-2):
        # Learning Rate
        self.learningRate = learningRate

        # Input/Output Size
        self.inputSize = inputSize
        self.outputSize = outputSize

        # Generate Weights
        self.WH = np.random.randn(self.inputSize, self.outputSize)

        # Generate Biases
        self.bh = np.zeros((self.inputSize, self.outputSize))

        # Cache for Adam Optimizer
        self.WHm = np.zeros_like(self.WH)
        self.WHv = np.zeros_like(self.WH)

        # Setup Activation Function
        self.activation, self.activationPrime = activations.get(activation)

    def loss(self, o, y):
        return np.mean(np.square(o - y), axis=-1)

    def lossPrime(self, o, y):
        return o - y

    def forward(self, X):
        o = self.activation(np.dot(X, self.WH))
        self.X = X
        self.o = o
        return o

    def backward(self, dY):
        dY = np.multiply(dY, self.activationPrime(self.o))
        dW = np.dot(self.X.T, dY)
        for param, m, v, delta in zip([self.WH], [self.WHm], [self.WHv], [dW]):
            m = 0.9 * m + 0.1 * delta
            v = 0.99 * v + 0.01 * (delta ** 2)
            param += -self.learningRate * m / (np.sqrt(v) + 1e-8)
        return dY
