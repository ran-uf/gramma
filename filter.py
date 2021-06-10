import numpy as np


def generate_data(length):
    a = np.random.random(length)
    flt = np.array([1, -0.8, 0.7])
    return a, np.convolve(a, flt)[:-2]
