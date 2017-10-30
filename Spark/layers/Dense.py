from .Layer import Layer
from ..activations import Activations

import numpy as np

class Dense(Layer):
    def __init__(self, inputSize, outputSize, activation="Linear"):
        # Generate Weights
        self.W = np.random.randn(inputSize, outputSize)

        # Generate Biases
        self.b = np.zeros((1, outputSize))

        # Setup Activation Function
        self.activation, self.activationPrime = Activations(activation)

    def addOptimizer(self, learningRate, optimizer):
        self.optimizer = optimizer(learningRate)

    def forward(self, X):
        self.X = X
        self.o = self.activation(np.dot(X, self.W) + self.b)
        return self.o

    def backward(self, gradient):
        X = self.X
        XShape = X.shape

        W = self.W
        WShape = W.shape

        b = self.b

        dO = np.multiply(gradient, self.activationPrime(self.o))
        dW = np.add(np.dot(X, np.ones(WShape)), np.dot(np.zeros(XShape), W))
        dB = np.add(dW, np.ones(b.shape))

        optimizer = self.optimizer
        self.W = optimizer.optimize(W, np.multiply(dO, dW))
        self.b = optimizer.optimize(b, np.multiply(dO, dB))

        return np.add(np.dot(X, np.zeros(WShape)), np.dot(np.ones(XShape), W))
