import numpy as np

class ADNode(object):
    def __init__(self, name, value):
        self.name = name
        self.value = value
        self.gradient = None
        self.parent = None

    def __str__(self):
        return str(self.value)

    def __add__(self, item):
        if type(item) is int:
            node = ADNode("Output", 0)
            operation = AddConstant(node, [self, item])
            node.operation = operation
            node.value = operation.compute()

            self.parent = node

            return node

        elif type(item) is ADNode:
            node = ADNode("Output", 0)
            operation = AddNode(node, [self, item])
            node.operation = operation
            node.value = operation.compute()

            self.parent = node
            item.parent = node

            return node

    def __sub__(self, item):
        if type(item) is int:
            node = ADNode("Output", 0)
            operation = SubtractConstant(node, [self, item])
            node.operation = operation
            node.value = operation.compute()

            self.parent = node

            return node

        elif type(item) is ADNode:
            node = ADNode("Output", 0)
            operation = SubtractNode(node, [self, item])
            node.operation = operation
            node.value = operation.compute()

            self.parent = node
            item.parent = node

            return node

    def __mul__(self, item):
        if type(item) is int:
            node = ADNode("Output", 0)
            operation = MultiplyConstant(node, [self, item])
            node.operation = operation
            node.value = operation.compute()

            self.parent = node

            return node

        if type(item) is ADNode:
            node = ADNode("Output", 0)
            operation = MultiplyNode(node, [self, item])
            node.operation = operation
            node.value = operation.compute()

            self.parent = node
            item.parent = node

            return node

    def __dot__(self, item):
        node = ADNode("Output", 0)
        operation = DotNode(node, [self, item])
        node.operation = operation
        node.value = operation.compute()

        self.parent = node
        item.parent = node

        return node

    def toGraph(self, name=None, level=0):
        if name == None:
            name = self.name
        nextLevel = level + 1
        indent = "    |" * level + "    "
        graph = "\x1b[34m" + self.operation.__class__.__name__ + "\x1b[0m \"" + name + "\""

        for node in self.operation.inputs:
            if type(node) is ADNode and node != self:
                graph += "\n" + indent + "| " + node.toGraph(level=nextLevel)
            elif type(node) is int:
                graph += "\n" + indent + "| " + str(node)

        return graph

class DefaultOperation(object):
    def __init__(self, inputs):
        self.inputs = inputs
        self.input = inputs[0]

    def compute(self):
        return self.input

    def gradient(self, node):
        return None

class AddConstant(object):
    def __init__(self, node, inputs):
        self.node = node
        self.inputs = inputs
        self.base = inputs[0]
        self.constant = inputs[1]

    def compute(self):
        output = np.add(self.base.value, self.constant)
        self.node.value = output
        return output

    def gradient(self):
        elements = []
        print self.node.gradient
        for child in self.inputs:
            print child
        # print np.sum(np.multiply() for child in self.inputs)
        return "HI"

class AddNode(object):
    def __init__(self, node, inputs):
        self.node = node
        self.inputs = inputs
        self.base = inputs[0]
        self.nodeToAdd = inputs[1]

    def compute(self):
        output = np.add(self.base.value, self.nodeToAdd.value)
        self.node.value = output
        return output

    def gradient(self):
        pass

class SubtractConstant(object):
    def __init__(self, node, inputs):
        self.node = node
        self.inputs = inputs
        self.base = inputs[0]
        self.constant = inputs[1]

    def compute(self):
        output = np.subtract(self.base.value, self.constant)
        self.node.value = output
        return output

    def gradient(self, node):
        base = self.base

        grad = None

        if base.gradient is None:
            if base == node:
                grad = np.ones(base.value.shape, dtype=float)
            else:
                grad = np.zeros(base.value.shape, dtype=float)
        else:
            grad = base.gradient

        self.node.gradient = grad
        return grad

class SubtractNode(object):
    def __init__(self, node, inputs):
        self.node = node
        self.inputs = inputs
        self.base = inputs[0]
        self.nodeToSubtract = inputs[1]

    def compute(self):
        output = np.subtract(self.base.value, self.nodeToSubtract.value)
        self.node.value = output
        return output

    def gradient(self, node):
        base = self.base
        nodeToSubtract = self.nodeToSubtract

        baseGrad = base.gradient
        nodeToSubtractGrad = nodeToSubtract.gradient

        grad = None

        if baseGrad is None:
            if base == node:
                baseGrad = np.ones(base.value.shape, dtype=float)
            else:
                baseGrad = np.zeros(base.value.shape, dtype=float)

        if nodeToSubtractGrad is None:
            if nodeToSubtract == node:
                nodeToSubtractGrad = np.ones(nodeToSubtract.value.shape, dtype=float)
            else:
                nodeToSubtractGrad = np.zeros(nodeToSubtract.value.shape, dtype=float)

        grad = np.subtract(baseGrad, nodeToSubtractGrad)
        self.node.gradient = grad
        return grad

class MultiplyConstant(object):
    def __init__(self, node, inputs):
        self.node = node
        self.inputs = inputs
        self.base = inputs[0]
        self.constant = inputs[1]

    def compute(self):
        output = np.multiply(self.base.value, self.constant)
        self.node.value = output
        return output

    def gradient(self, node):
        base = self.base
        constant = self.constant

        baseGrad = base.gradient

        grad = None

        if baseGrad is None:
            if base == node:
                baseGrad = np.ones(base.value.shape, dtype=float)
            else:
                baseGrad = np.zeros(base.value.shape, dtype=float)

        grad = np.multiply(baseGrad, constant)
        self.node.gradient = grad
        return grad

class MultiplyNode(object):
    def __init__(self, node, inputs):
        self.node = node
        self.inputs = inputs
        self.base = inputs[0]
        self.nodeToMultiply = inputs[1]

    def compute(self):
        output = np.multiply(self.base.value, self.nodeToMultiply.value)
        self.node.value = output
        return output

    def gradient(self, node):
        base = self.base
        nodeToMultiply = self.nodeToMultiply

        baseGrad = base.gradient
        nodeToMultiplyGrad = nodeToMultiply.gradient

        grad = None

        if baseGrad is None:
            if base == node:
                baseGrad = np.ones(base.value.shape, dtype=float)
            else:
                baseGrad = np.zeros(base.value.shape, dtype=float)

        if nodeToMultiplyGrad is None:
            if nodeToMultiply == node:
                nodeToMultiplyGrad = np.ones(nodeToMultiply.value.shape, dtype=float)
            else:
                nodeToMultiplyGrad = np.zeros(nodeToMultiply.value.shape, dtype=float)

        grad = np.add(np.multiply(base.value, nodeToMultiplyGrad), np.multiply(baseGrad, nodeToMultiply.value))
        self.node.gradient = grad
        return grad

class DotNode(object):
    def __init__(self, node, inputs):
        self.node = node
        self.inputs = inputs
        self.base = inputs[0]
        self.nodeToDot = inputs[1]

    def compute(self):
        output = np.dot(self.base.value, self.nodeToDot.value)
        self.node.value = output
        return output

    def gradient(self, node):
        grad = None

        if nodeGradient is not None:
            self.node.parent.operation.gradient(self.node)
            grad = np.dot(self.base.value.T, self.node.parent.gradient)

        self.node.gradient = grad
        return grad

class CompiledFunction(object):
    def __init__(self, inputs, output):
        self.inputs = {}
        self.output = output

        for inputItem in inputs:
            self.inputs[inputItem.name] = inputItem

    def __call__(self, *inputList):
        graph = self.output
        inputs = self.inputs

        for i, inputItem in enumerate(inputs):
            inputListItem = inputList[i]
            if type(inputListItem) is ADNode:
                inputs[inputItem].value = inputList[i].value
            else:
                inputs[inputItem].value = inputList[i]

        def computeGate(node):
            children = node.operation.inputs
            for child in children:
                if type(child) is ADNode and child != node and type(child.operation) is not DefaultOperation:
                    computeGate(child)
            node.operation.compute()

        children = graph.operation.inputs
        for child in children:
            if type(child) is ADNode and child != graph and type(child.operation) is not DefaultOperation:
                computeGate(child)

        return graph.operation.compute()

def variable(name, value):
    node = ADNode(name, value)
    node.operation = DefaultOperation([node])
    return node

def gradient(outputFunction, node):
    respectNode = [node]
    d = None

    def computeParent(node):
        if node.parent != None:
            return computeParent(node.parent)
        else:
            return node

    # def computeChildren(node):
    #     d[0] *= node.operation.gradient(respectNode[0])
    #     children = node.operation.inputs
    #     for child in children:
    #         if type(child) is ADNode and child != node and type(child.operation) is not DefaultOperation:
    #             computeChildren(node)

    parent = computeParent(node)
    # parent.gradient = np.ones(parent.value.shape)
    # d = parent.operation.gradient()

    return d

def function(inputs, output):
    return CompiledFunction(inputs, output)
