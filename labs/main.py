import numpy as np
import pandas as pd

np.random.seed(0)

x1 = np.random.randint(10, size=6)
print(x1)
# print(x1)
# print("-----")
# x2 = np.random.randint(10, size=(3,4))
# print(x2)
# print("-----")


# X = np.random.randint(10, size=15)
# Y = np.arange(0, 15)
# X1 = X*Y
# c = np.sum(X1)
# print(c)

# X = np.random.randint(10, size=15)
# m = np.amax(X)
# n = np.amin(X)
#
# X[m],X[n] = np.amin(X), np.amin(X)
# print(m)
# print(n)


# data = pd.Series([0.25, 0.5, 0.75, 1.0])
# print(data)
# print(data.values)
# print(data.index)

# x3 = np.random.randint(10, size=(3, 4, 5))
# print(x3)

print("-----")

# print(x3[0][2][4])
# print(x3.shape)
# print(x3.size)

# y1 = np.random.randint(10, size=6)
#
# xy = np.concatenate([x1, y1])
#
# z1, z2, z3 = np.split(xy, [3, 5])
#
# print(z1)
# print(z2)
# print(z3)
print(np.subtract(x1, 2))