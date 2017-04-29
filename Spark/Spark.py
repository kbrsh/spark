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
        self.learningRate = learningRate

    def run(self, epochs=10):
        inputs = self.X
        outputs = self.y

        lastLayer = self.layers[len(self.layers) - 1]

        for epoch in xrange(epochs):
            loss = 0

            lastInput = inputs
            y = outputs

            # Forward Propagate Layers
            for layer in self.layers:
                lastInput = layer.forward(lastInput)

            # Obtain Loss
            loss += self.loss(lastInput, y)

            # Backward Propagate Layers
            dY = self.lossPrime(lastInput, y)
            gradients = []

            for layer in reversed(self.layers):
                dY, grad = layer.backward(dY)
                gradients.append(grad)

            # Perform Parameter Update
            for layer, grad in zip(reversed(self.layers), gradients):
                params, ms, vs = layer.getParams()
                for param, m, v, delta in zip(params, ms, vs, grad):
                    m = 0.9 * m + 0.1 * delta
                    v = 0.99 * v + 0.01 * (delta ** 2)
                    param += -self.learningRate * m / (np.sqrt(v) + 1e-8)



            print 'Epoch: ' + str(epoch)
            print 'Loss: {0:.20f}'.format(np.mean(loss))
            print ''


    def predict(self, y):
        for layer in self.layers:
            y = layer.forward(y)

        return y
