import numpy as np

arr = np.zeros(20 * 20).reshape(20, 20, 1)
arr[:, 0] = 2
arr[:, -1] = 2
print(arr.reshape((20, 20)))
