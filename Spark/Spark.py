import numpy as np

class Spark(object):
    def __init__(self):
        # Layers
        self.layers = []

    def add(self, layer):
        self.layers.append(layer)

    def train(self, X, y):
        # Setup inputs
        self.inputSize = X[0].shape[0]
        self.X = X

        # Setup outputs
        self.outputSize = y[0].shape[0]
        self.y = y

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

            loss += lastLayer.loss(lastInput, y)

            dY = lastLayer.lossPrime(lastInput, y)
            for layer in reversed(self.layers):
                dY = layer.backward(dY)

            print 'Epoch: ' + str(epoch)
            print 'Loss: {0:.20f}'.format(np.mean(loss))
            print ''


    def predict(self, y):
        for layer in self.layers:
            y = layer.forward(y)

        return y
