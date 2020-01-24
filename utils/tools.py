import numpy as np


def x_y_meshgrid(x_row, y_col):
    x = np.arange(0, x_row)
    y = np.arange(0, y_col)
    X, Y = np.meshgrid(x, y)
    X = X.flatten()
    Y = Y.flatten()
    return X, Y