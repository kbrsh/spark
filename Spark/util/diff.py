import numpy as np

outputVariableCount = 0
def outputVariable():
    global outputVariableCount
    outputVariableCount += 1
    return "o" + str(outputVariableCount)

def ADGradientArray(arr):
    return "sp.array(" + np.array2string(arr, separator=", ") + ")"

def ADGradientOperation(op, a, b):
    if len(a) > 9 and a[0:8] == "sp.zeros":
        if op == "add":
            return b
        elif op == "multiply":
            return a

    if len(a) > 8 and a[0:7] == "sp.ones":
        if op == "multiply":
            return b

    if len(b) > 9 and b[0:8] == "sp.zeros":
        if op == "add" or op == "subtract":
            return a
        elif op == "multiply":
            return b

    if len(b) > 8 and b[0:7] == "sp.ones":
        if op == "multiply":
            return a

    return "sp." + op + "(" + a + ", " + b + ")"

class ADNode(object):
    def __init__(self, name, value):
        self.name = name
        self.value = value

    def __str__(self):
        return str(self.value)

    def compute(self):
        operation = self.operation
        inputs = operation.inputs

        for node in inputs:
            if type(node) is ADNode:
                node.compute()

        newValue = operation.compute()
        self.value = newValue
        return newValue

    def add(self, item):
        if type(item) is ADNode:
            node = ADNode(outputVariable(), 0)
            operation = AddNode(node, [self, item])
            node.operation = operation
            node.value = operation.compute()

            return node
        else:
            node = ADNode(outputVariable(), 0)
            operation = AddConstant(node, [self, item])
            node.operation = operation
            node.value = operation.compute()

            return node

    def subtract(self, item):
        if type(item) is ADNode:
            node = ADNode(outputVariable(), 0)
            operation = SubtractNode(node, [self, item])
            node.operation = operation
            node.value = operation.compute()

            return node
        else:
            node = ADNode(outputVariable(), 0)
            operation = SubtractConstant(node, [self, item])
            node.operation = operation
            node.value = operation.compute()

            return node

    def multiply(self, item):
        if type(item) is ADNode:
            node = ADNode(outputVariable(), 0)
            operation = MultiplyNode(node, [self, item])
            node.operation = operation
            node.value = operation.compute()

            return node
        else:
            node = ADNode(outputVariable(), 0)
            operation = MultiplyConstant(node, [self, item])
            node.operation = operation
            node.value = operation.compute()

            return node

    def dot(self, item):
        if type(item) is ADNode:
            node = ADNode(outputVariable(), 0)
            operation = DotNode(node, [self, item])
            node.operation = operation
            node.value = operation.compute()

            return node
        else:
            node = ADNode(outputVariable(), 0)
            operation = DotConstant(node, [self, item])
            node.operation = operation
            node.value = operation.compute()

            return node

    def graph(self, level=0):
        name = self.name
        nextLevel = level + 1
        indent = "    |" * level + "    "
        graph = "\x1b[34m" + self.operation.__class__.__name__ + "\x1b[0m \"" + name + "\""

        for node in self.operation.inputs:
            if type(node) is ADNode:
                graph += "\n" + indent + "| " + node.graph(level=nextLevel)
            else:
                graph += "\n" + indent + "| \x1b[34mConstant\x1b[0m " + str(node)

        return graph

class Operation(object):
    def compute(self):
        pass
    def gradient(self, respect):
        pass

class Variable(Operation):
    def __init__(self, node):
        self.node = node
        self.inputs = []

    def compute(self):
        return self.node.value

    def gradient(self, respect):
        node = self.node
        shape = str(node.value.shape)
        if respect == node:
            return "sp.ones(" + shape + ")"
        else:
            return "sp.zeros(" + shape + ")"

class AddConstant(Operation):
    def __init__(self, node, inputs):
        self.node = node
        self.inputs = inputs

    def compute(self):
        inputs = self.inputs
        return np.add(inputs[0].value, inputs[1])

    def gradient(self, respect):
        return self.inputs[0].operation.gradient(respect)

class AddNode(Operation):
    def __init__(self, node, inputs):
        self.node = node
        self.inputs = inputs

    def compute(self):
        inputs = self.inputs
        return np.add(inputs[0].value, inputs[1].value)

    def gradient(self, respect):
        inputs = self.inputs
        return ADGradientOperation("add", inputs[0].operation.gradient(respect), inputs[1].operation.gradient(respect))

class SubtractConstant(Operation):
    def __init__(self, node, inputs):
        self.node = node
        self.inputs = inputs

    def compute(self):
        inputs = self.inputs
        return np.subtract(inputs[0].value, inputs[1])

    def gradient(self, respect):
        return self.inputs[0].operation.gradient(respect)

class SubtractNode(Operation):
    def __init__(self, node, inputs):
        self.node = node
        self.inputs = inputs

    def compute(self):
        inputs = self.inputs
        return np.subtract(inputs[0].value, inputs[1].value)

    def gradient(self, respect):
        inputs = self.inputs
        return ADGradientOperation("subtract", inputs[0].operation.gradient(respect), inputs[1].operation.gradient(respect))

class MultiplyConstant(Operation):
    def __init__(self, node, inputs):
        self.node = node
        self.inputs = inputs

    def compute(self):
        inputs = self.inputs
        return np.multiply(inputs[0].value, inputs[1])

    def gradient(self, respect):
        inputs = self.inputs
        return ADGradientOperation("multiply", inputs[0].operation.gradient(respect), ADGradientArray(inputs[1]))

class MultiplyNode(Operation):
    def __init__(self, node, inputs):
        self.node = node
        self.inputs = inputs

    def compute(self):
        inputs = self.inputs
        return np.multiply(inputs[0].value, inputs[1].value)

    def gradient(self, respect):
        inputs = self.inputs
        return ADGradientOperation("add",
            ADGradientOperation("multiply", inputs[0].name, inputs[1].operation.gradient(respect)),
            ADGradientOperation("multiply", inputs[0].operation.gradient(respect), inputs[1].name)
        )

class DotConstant(Operation):
    def __init__(self, node, inputs):
        self.node = node
        self.inputs = inputs

    def compute(self):
        inputs = self.inputs
        return np.dot(inputs[0].value, inputs[1])

    def gradient(self, respect):
        inputs = self.inputs
        return ADGradientOperation("dot", inputs[0].operation.gradient(respect), ADGradientArray(inputs[1]))

class DotNode(Operation):
    def __init__(self, node, inputs):
        self.node = node
        self.inputs = inputs

    def compute(self):
        inputs = self.inputs
        return np.dot(inputs[0].value, inputs[1].value)

    def gradient(self, respect):
        inputs = self.inputs
        return ADGradientOperation("add",
            ADGradientOperation("dot", inputs[0].name, inputs[1].operation.gradient(respect)),
            ADGradientOperation("dot", inputs[0].operation.gradient(respect), inputs[1].name)
        )

def variable(name, value):
    node = ADNode(name, value)
    node.operation = Variable(node)
    return node

def gradient(output, respect):
    return output.operation.gradient(respect)
