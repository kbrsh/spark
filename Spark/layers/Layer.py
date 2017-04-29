class Layer(object):
    def forward(self, X):
        pass

    def backward(self, dO):
        pass

    def getParams(self):
        return [], [], []
