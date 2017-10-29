import numpy as np
from .losses import losses

class Spark(object):
    def __init__(self, inputs, outputs, learningRate=1e-2, loss="meanSquared", layers=[]):
        # Inputs
        self.X = inputs

        # Outputs
        self.y = outputs

        # Learning Rate
        self.learningRate = learningRate

        # Loss
        self.loss, self.lossPrime = losses(loss)

        # Layers
        self.layers = layers

    def run(self, epochs=10):
        inputs = self.X
        outputs = self.y

        for epoch in range(epochs):
            lastInput = inputs
            y = outputs

            # Forward Propagate Layers
            for layer in self.layers:
                lastInput = layer.forward(lastInput)

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



            print("Epoch: " + str(epoch))
            print("Loss: " + str(self.loss(lastInput, y)))
            print()

    def predict(self, y):
        for layer in self.layers:
            y = layer.forward(y)

        return y
