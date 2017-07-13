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
            operation = AddConstant([self, item])
            node.operation = operation
            node.value = operation.compute()

            self.parents.append(node)

            return node

        elif type(item) is AGNode:
            node = AGNode("Output", 0)
            operation = AddNode([self, item])
            node.operation = operation
            node.value = operation.compute()

            self.parents.append(node)
            item.parents.append(node)

            return node

    def __mul__(self, item):
        if type(item) is int:
            node = AGNode("Output", 0)
            operation = MultiplyConstant([self, item])
            node.operation = operation
            node.value = operation.compute()

            self.parents.append(node)

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

    def gradient(self):
        return None

class AddConstant(object):
    def __init__(self, inputs):
        self.inputs = inputs
        self.base = inputs[0]
        self.constant = inputs[1]

    def compute(self):
        return self.base.value + self.constant

    def gradient(self):
        return 1

class AddNode(object):
    def __init__(self, inputs):
        self.inputs = inputs
        self.base = inputs[0]
        self.node = inputs[1]

    def compute(self):
        return self.base.value + self.node.value

    def gradient(self):
        return 1

class MultiplyConstant(object):
    def __init__(self, inputs):
        self.inputs = inputs
        self.base = inputs[0]
        self.constant = inputs[1]

    def compute(self):
        return self.base.value * self.constant

    def gradient(self):
        return self.constant

class CompiledFunction(object):
    def __init__(self, inputs, output):
        self.inputs = {}
        self.output = output

        for inputItem in inputs:
            self.inputs[inputItem.name] = inputItem

    def __call__(self, *inputList):
        graph = self.output
        inputs = self.inputs
        output = [0]

        for i, inputItem in enumerate(inputs):
            inputs[inputItem].value = inputList[i]

        def computeGate(node):
            output[0] += node.operation.compute()
            children = node.operation.inputs
            for child in children:
                if type(child) is AGNode and child != node and type(child.operation) is not DefaultOperation:
                    computeGate(child)

        computeGate(graph)

        return output[0]

def variable(name, value):
    node = AGNode(name, value)
    node.operation = DefaultOperation([node])
    return node

def gradient(output, node):
    nodeGradient = node.operation.gradient()

    d = [1]

    def compute(_node):
        changed = False
        _d = 0
        for parent in _node.parents:
            if changed == False:
                changed = True
            _d += parent.operation.gradient()
            compute(parent)

        if changed == True:
            d[0] *= _d

    compute(node)

    return d[0]

def function(inputs, output):
    return CompiledFunction(inputs, output)
