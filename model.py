import numpy as np


def model_fun(x, p1, p2):
    return p1 * np.cos(p2 * x) + p2 * np.sin(p1 * x)
