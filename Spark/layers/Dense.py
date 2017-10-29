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

    def backward(self, dO):
        X = self.X
        W = self.W
        gradient = np.multiply(np.multiply(dO, self.activationPrime(self.o)), np.add(np.dot(X, np.ones(W.shape)), np.dot(np.zeros(X.shape), W)))
        self.W = self.optimizer.optimize(W, gradient)
        return gradient
