class Optimizer(object):
    def optimize(self, weight, gradient):
        pass

class VanillaOptimizer(Optimizer):
    def __init__(self, learningRate):
        # Learning Rate
        self.learningRate = learningRate

    def optimize(self, weight, gradient):
        return weight - (self.learningRate * gradient)

allOptimizers = globals()

def Optimizers(name):
    return allOptimizers["".join(name.title().split(" ")).title() + "Optimizer"]
