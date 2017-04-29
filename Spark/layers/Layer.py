class Layer(object):
    def forward(self, X):
        pass

    def backward(self, dY):
        pass

    def getParams(self):
        return [], [], []
