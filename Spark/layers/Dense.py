from . import Layer
from .. import activations

import numpy as np

class Dense(Layer.Layer):
    def __init__(self, inputSize, outputSize, activation=None):
        # Input/Output Size
        self.inputSize = inputSize
        self.outputSize = outputSize

        # Generate Weights
        self.WH = np.random.randn(self.inputSize, self.outputSize) * 0.1

        # Generate Biases
        self.bh = np.zeros((1, self.outputSize))

        # Cache for Adam Optimizer
        self.WHm = np.zeros_like(self.WH)
        self.WHv = np.zeros_like(self.WH)

        self.bhm = np.zeros_like(self.bh)
        self.bhv = np.zeros_like(self.bh)

        # Setup Activation Function
        self.activation, self.activationPrime = activations.get(activation)

    def forward(self, X):
        self.X = X
        self.o = self.activation(np.dot(X, self.WH) + self.bh)
        return self.o

    def backward(self, dO):
        # Back Propagate into Dot Product
        dY = np.multiply(dO, self.activationPrime(self.o))

        # Back Propagate into Bias
        dbh = np.sum(dO, axis=0, keepdims=True)

        # Back Propagate into Weights
        dW = np.dot(self.X.T, dY)

        # Back Propagate into Inputs
        dX = np.dot(dY, self.WH.T)

        return dX, [dW, dbh]

    def getParams(self):
        return [self.WH, self.bh], [self.WHm, self.bhm], [self.WHv, self.bhv]
