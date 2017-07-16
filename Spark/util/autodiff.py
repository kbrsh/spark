import numpy as np

class AGNode(object):
    def __init__(self, name, value):
        self.name = name
        self.value = value
        self.gradient = None
        self.parent = None

    def __str__(self):
        return str(self.value)

    def __add__(self, item):
        if type(item) is int:
            node = AGNode("Output", 0)
            operation = AddConstant(node, [self, item])
            node.operation = operation
            node.value = operation.compute()

            self.parent = node

            return node

        elif type(item) is AGNode:
            node = AGNode("Output", 0)
            operation = AddNode(node, [self, item])
            node.operation = operation
            node.value = operation.compute()

            self.parent = node
            item.parent = node

            return node

    def __sub__(self, item):
        if type(item) is int:
            node = AGNode("Output", 0)
            operation = SubtractConstant(node, [self, item])
            node.operation = operation
            node.value = operation.compute()

            self.parent = node

            return node

        elif type(item) is AGNode:
            node = AGNode("Output", 0)
            operation = SubtractNode(node, [self, item])
            node.operation = operation
            node.value = operation.compute()

            self.parent = node
            item.parent = node

            return node

    def __mul__(self, item):
        if type(item) is int:
            node = AGNode("Output", 0)
            operation = MultiplyConstant(node, [self, item])
            node.operation = operation
            node.value = operation.compute()

            self.parent = node

            return node

        if type(item) is AGNode:
            node = AGNode("Output", 0)
            operation = MultiplyNode(node, [self, item])
            node.operation = operation
            node.value = operation.compute()

            self.parent = node
            item.parent = node

            return node

    def __dot__(self, item):
        node = AGNode("Output", 0)
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
        indent = "  |" * level + "  "
        graph = "\x1b[34m" + self.operation.__class__.__name__ + "\x1b[0m \"" + name + "\""

        for node in self.operation.inputs:
            if type(node) is AGNode and node != self:
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

    def gradient(self, node):
        base = self.base
        nodeToAdd = self.nodeToAdd

        baseGrad = base.gradient
        nodeToAddGrad = nodeToAdd.gradient

        grad = None

        if baseGrad is None:
            if base == node:
                baseGrad = np.ones(base.value.shape, dtype=float)
            else:
                baseGrad = np.zeros(base.value.shape, dtype=float)

        if nodeToAddGrad is None:
            if nodeToAdd == node:
                nodeToAddGrad = np.ones(nodeToAdd.value.shape, dtype=float)
            else:
                nodeToAddGrad = np.zeros(nodeToAdd.value.shape, dtype=float)

        grad = np.add(baseGrad, nodeToAddGrad)
        self.node.gradient = grad
        return grad

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
        return np.zeros(self.base.value.shape, dtype=float)

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
        return np.full(self.base.value.shape, self.constant, dtype=float)

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
        if node == self.base:
            return self.nodeToMultiply.value
        else:
            return self.base.value

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
        return np.dot(self.base.value.T, self.node.parent.operation.compute())

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
            if type(inputListItem) is AGNode:
                inputs[inputItem].value = inputList[i].value
            else:
                inputs[inputItem].value = inputList[i]

        def computeGate(node):
            children = node.operation.inputs
            for child in children:
                if type(child) is AGNode and child != node and type(child.operation) is not DefaultOperation:
                    computeGate(child)
            node.operation.compute()

        children = graph.operation.inputs
        for child in children:
            if type(child) is AGNode and child != graph and type(child.operation) is not DefaultOperation:
                computeGate(child)

        return graph.operation.compute()

def variable(name, value):
    node = AGNode(name, value)
    node.operation = DefaultOperation([node])
    return node

def gradient(outputFunction, node):
    respectNode = [node]
    output = [outputFunction.output]
    d = 0

    def computeChildren(_node):
        children = _node.operation.inputs
        for child in children:
            if type(child) is AGNode and child != _node and type(child.operation) is not DefaultOperation:
                computeGate(child)
        _node.operation.gradient(respectNode[0])

    def computeParent(_node):
        if _node.parent != None:
            return computeParent(_node.parent)
        else:
            return _node

    parent = computeParent(node)
    children = parent.operation.inputs
    for child in children:
        if type(child) is AGNode and child != parent and type(child.operation) is not DefaultOperation:
            computeChildren(child)

    return parent.operation.gradient(node)

def function(inputs, output):
    return CompiledFunction(inputs, output)
