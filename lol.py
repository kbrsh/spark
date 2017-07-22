import numpy as np
from theano import *
import theano.tensor as T

x = T.dscalar('x')
w = T.dscalar('w')
y = T.dot(x, w)
fn = theano.function([x, w], y)
lol = fn([[1, 2, 3],
          [1, 2, 3]], [[2],
                       [2],
                       [2]])
print lol

print T.grad(y, w)
# d = np.dot(np.dot(arr, w).T, arr)
# print d
#
# w += d
# o = np.dot(arr, w)
# print o
