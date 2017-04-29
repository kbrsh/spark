import numpy as np
from . import losses

class Spark(object):
    def __init__(self):
        # Layers
        self.layers = []

        # Loss
        self.loss = None
        self.lossPrime = None

    def add(self, layer):
        self.layers.append(layer)

    def train(self, X, y):
        # Setup inputs
        self.inputSize = X[0].shape[0]
        self.X = X

        # Setup outputs
        self.outputSize = y[0].shape[0]
        self.y = y

    def build(self, learningRate=1e-2, loss='meanSquared'):
        a, b = losses.get(loss)
        self.loss, self.lossPrime = losses.get(loss)

        for layer in self.layers:
            layer.learningRate = learningRate

    def run(self, epochs=10):
        inputs = self.X
        outputs = self.y

        lastLayer = self.layers[len(self.layers) - 1]

        for epoch in xrange(epochs):
            loss = 0

            lastInput = inputs
            y = outputs

            for layer in self.layers:
                lastInput = layer.forward(lastInput)

            loss += self.loss(lastInput, y)

            dY = self.lossPrime(lastInput, y)
            for layer in reversed(self.layers):
                dY = layer.backward(dY)

            print 'Epoch: ' + str(epoch)
            print 'Loss: {0:.20f}'.format(np.mean(loss))
            print ''


    def predict(self, y):
        for layer in self.layers:
            y = layer.forward(y)

        return y
