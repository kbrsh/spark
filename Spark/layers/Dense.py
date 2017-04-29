from . import Layer
from .. import activations

import numpy as np

class Dense(Layer.Layer):
    def __init__(self, inputSize, outputSize, activation=None):
        # Learning Rate
        self.learningRate = 0

        # Input/Output Size
        self.inputSize = inputSize
        self.outputSize = outputSize

        # Generate Weights
        self.WH = np.random.randn(self.inputSize, self.outputSize) * 0.1

        # Generate Biases
        self.bh = np.zeros(self.outputSize)

        # Cache for Adam Optimizer
        self.WHm = np.zeros_like(self.WH)
        self.WHv = np.zeros_like(self.WH)

        self.bhm = np.zeros_like(self.bh)
        self.bhv = np.zeros_like(self.bh)

        # Setup Activation Function
        self.activation, self.activationPrime = activations.get(activation)

    def forward(self, X):
        o = self.activation(np.dot(X, self.WH) + self.bh)
        self.X = X
        self.o = o
        return o

    def backward(self, dY):
        dbh = np.sum(dY)
        dY = np.multiply(dY, self.activationPrime(self.o))
        dW = np.dot(self.X.T, dY)
        return dY, [dW, dbh]

    def getParams(self):
        return [self.WH, self.bh], [self.WHm, self.bhm], [self.WHv, self.bhv]
