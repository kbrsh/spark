import numpy as np
import Spark.util as sp

# X = sp.variable('X', np.array([[1, 1]]))
# w = sp.variable('w', np.random.random((2, 1)))
# y = sp.variable('y', np.array([[1]]))
# o = sp.dot(X, w)
# l = sp.subtract(o, y)
# fit = sp.function([X, w], o)
#
#
#
# print
# print l.toGraph()
# print
# print fit(X.value, w.value)
# print
# print sp.gradient(fit, w)

addBase = sp.variable("addBase", np.array([[1, 1]]))
addS = sp.variable("addS", np.array([[2], [1]]))
output = sp.add(addBase, 10)
fn = sp.function([], output)

print
print output.toGraph()
print
print fn()
# addS.value = np.array([[1, 1]])
# print fn()
print
print sp.gradient(fn, addBase)
