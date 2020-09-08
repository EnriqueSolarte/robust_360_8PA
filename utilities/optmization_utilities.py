import numpy as np


def normalizer_s_k(x, s, k):
    n_matrix = np.eye(3) * s
    n_matrix[2, 2] = k
    return n_matrix @ x, n_matrix


def normalizer_s(x, s):
    n_matrix = np.eye(3) * s
    n_matrix[2, 2] = 10 / s
    return n_matrix @ x, n_matrix


def normalizer_3dv2020(x, s, k):
    x_mean = np.mean(x, axis=1)
    t = np.array([[s, 0, -s * x_mean[0]],
                  [0, s, -s * x_mean[1]],
                  [0, 0, k ** abs(1 - x_mean[2])]])
    # t = np.array([[s, 0, 0], [0, s, 0], [0, 0, k]])
    x_norm = np.dot(t, x)
    return x_norm, t
