import numpy as np
import util as sp

class AGNode(object):
    def __init__(self, name, value):
        self.name = name
        self.value = value
        self.parents = []

    def __str__(self):
        return str(self.value)

    def __radd__(self, item):
        pass

    def __add__(self, item):
        if type(item) is int:
            node = AGNode("Output", 0)
            operation = AddConstant(node, [self, item])
            node.operation = operation
            node.value = operation.compute()

            self.parents.append(node)

            return node

        elif type(item) is AGNode:
            node = AGNode("Output", 0)
            operation = AddNode(node, [self, item])
            node.operation = operation
            node.value = operation.compute()

            self.parents.append(node)
            item.parents.append(node)

            return node

    def __mul__(self, item):
        if type(item) is int:
            node = AGNode("Output", 0)
            operation = MultiplyConstant(node, [self, item])
            node.operation = operation
            node.value = operation.compute()

            self.parents.append(node)

            return node

        if type(item) is AGNode:
            node = AGNode("Output", 0)
            operation = MultiplyNode(node, [self, item])
            node.operation = operation
            node.value = operation.compute()

            self.parents.append(node)
            item.parents.append(node)

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
        output = sp.add(self.base.value, self.constant)
        self.node.value = output
        return output

    def gradient(self, node):
        return sp.ones(self.base.value.shape, dtype=float)

class AddNode(object):
    def __init__(self, node, inputs):
        self.node = node
        self.inputs = inputs
        self.base = inputs[0]
        self.nodeToAdd = inputs[1]

    def compute(self):
        output = sp.add(self.base.value, self.nodeToAdd.value)
        self.node.value = output
        return output

    def gradient(self, node):
        return sp.ones(self.base.value.shape, dtype=float)

class MultiplyConstant(object):
    def __init__(self, node, inputs):
        self.node = node
        self.inputs = inputs
        self.base = inputs[0]
        self.constant = inputs[1]

    def compute(self):
        output = sp.multiply(self.base.value, self.constant)
        self.node.value = output
        return output

    def gradient(self, node):
        return sp.full(self.base.value.shape, self.constant, dtype=float)

class MultiplyNode(object):
    def __init__(self, node, inputs):
        self.node = node
        self.inputs = inputs
        self.base = inputs[0]
        self.nodeToMultiply = inputs[1]

    def compute(self):
        output = sp.multiply(self.base.value, self.nodeToMultiply.value)
        self.node.value = output
        return output

    def gradient(self, node):
        if node == self.base:
            return self.nodeToMultiply.value
        else:
            return self.base.value

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
    node = [node]
    output = [outputFunction.output]
    d = [1]

    def compute(_node):
        changed = False
        _d = 0
        for parent in _node.parents:
            if changed == False:
                changed = True
            _d += parent.operation.gradient(node[0])
            compute(parent)

        if changed == True:
            d[0] *= _d

    compute(node[0])

    return d[0]

def function(inputs, output):
    return CompiledFunction(inputs, output)
