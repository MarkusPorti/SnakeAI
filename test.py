import numpy as np

board = np.zeros((10, 10), dtype=int)
snake = np.array([(5, 5), (5, 6), (5, 7)])
board[snake[1:, 1], snake[1:, 0]] = 1
board[:, 0] = 1  # left wall
board[:, -1] = 1  # right wall
board[0, :] = 1  # top wall
board[-1, :] = 1  # bottom wall
print(board)



arr = np.zeros(20 * 20).reshape(20, 20, 1)
arr[:, 0] = 2
arr[:, -1] = 2
print(arr.reshape((20, 20)))
