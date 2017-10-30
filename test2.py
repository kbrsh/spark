from Spark import Spark
from Spark.layers import Dense
import Spark.util as sp

# bot = Spark(
#     inputs=sp.array([
#         [1, 0],
#         [0, 1],
#         [1, 1],
#         [0, 0]
#     ]),
#     outputs=sp.array([
#         [1],
#         [1],
#         [0],
#         [0]
#     ]),
#     layers=[
#         Dense(2, 10),
#         Dense(10, 1)
#     ]
# )
#
# bot.run(epochs=3)
# print(bot.predict([[0, 0]]))

# X = sp.variable("X", sp.random(4, 10))
# w = sp.variable("w", sp.random(10, 1))
# b = sp.variable("b", sp.random(1, 1))
# o = sp.add(sp.dot(X, w), b)
# o = sp.dot(X, w)
# print(o)
# print(sp.gradient(o, w))

X = sp.variable("X", 15)
o = sp.add(X, 10)
print(o)
print(o.graph())
print(sp.gradient(o, X))








# import numpy as np
# import Spark.util as sp
#
#
# X = sp.variable("X", sp.array([[1, 0], [0, 1]]))
# w = sp.variable("w", sp.random(2, 1))
# y = sp.variable("y", sp.random(2, 1))
# l = sp.subtract(y, sp.dot(X, w))
# print()
# print(l.graph())
# print()
# print(l)
# print()
# print(sp.gradient(l, w))
# print()
# w.value -= 0.1 * np.subtract(np.zeros((2, 1)), np.add(np.dot(X.value, np.ones((2, 1))), np.dot(np.zeros((2, 2)), w.value)))
# w.value -= 0.1 * np.subtract(np.zeros((2, 1)), np.add(np.dot(X.value, np.ones((2, 1))), np.dot(np.zeros((2, 2)), w.value)))
# w.value -= 0.1 * np.subtract(np.zeros((2, 1)), np.add(np.dot(X.value, np.ones((2, 1))), np.dot(np.zeros((2, 2)), w.value)))
# w.value -= 0.1 * np.subtract(np.zeros((2, 1)), np.add(np.dot(X.value, np.ones((2, 1))), np.dot(np.zeros((2, 2)), w.value)))
# w.value -= 0.1 * np.subtract(np.zeros((2, 1)), np.add(np.dot(X.value, np.ones((2, 1))), np.dot(np.zeros((2, 2)), w.value)))
# l.compute()
# print(l.value)
#
#
#
# # # X = sp.variable('X', np.array([[1, 1]]))
# # # w = sp.variable('w', np.random.random((2, 1)))
# # # y = sp.variable('y', np.array([[1]]))
# # # o = sp.dot(X, w)
# # # l = sp.subtract(o, y)
# # # fit = sp.function([X, w], o)
# # #
# # #
# # #
# # # print
# # # print l.toGraph()
# # # print
# # # print fit(X.value, w.value)
# # # print
# # # print sp.gradient(fit, w)
# #
# # addBase = sp.variable("addBase", np.array([[1, 1]]))
# # addS = sp.variable("addS", np.array([[2], [1]]))
# # output = sp.add(sp.add(addBase, 10), 10)
# # fn = sp.function([], output)
# #
# # print
# # print output.toGraph()
# # print
# # print fn()
# # # addS.value = np.array([[1, 1]])
# # # print fn()
# # print
# # print sp.gradient(fn, addBase)
# b
