import numpy as np
from matplotlib import pyplot as plt

x = np.arange(0, 10.5, 0.5)
y = np.cos(x)
plt.scatter(x, y)
plt.show()