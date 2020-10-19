import numpy as np
from geometry_utilities import *


def normalizer_exp_s_k(x, s, k):
    n_matrix = np.eye(3) * (1 + np.exp(s))
    n_matrix[2, 2] = (1 + np.exp(k))
    return n_matrix @ x, n_matrix


def normalizer_s_k(x, s, k):
    n_matrix = np.eye(3) * s
    n_matrix[2, 2] = k
    return n_matrix @ x, n_matrix


def normalizer_Cxyz(x, Cxyz):
    n_matrix = np.eye(3)
    n_matrix[0, 0] = Cxyz[0]
    n_matrix[1, 1] = Cxyz[1]
    n_matrix[2, 2] = Cxyz[2]

    return n_matrix @ x, n_matrix


def normalizer_s(x, s):
    n_matrix = np.eye(3) * s
    n_matrix[2, 2] = 1 / s
    return n_matrix @ x, n_matrix


def normalizer_Hartley_isotropic(x):
    x_mean = np.mean(x, axis=1)
    x_std = np.linalg.norm(x - x_mean.reshape((3, -1)), axis=0)
    x_std = 0.5 * np.sum(x_std ** 2) / x_std.size
    x_std = 1 / np.sqrt(x_std)
    n_matrix = np.eye(3) * x_std
    n_matrix[0, 2] = -x_std * x_mean[0]
    n_matrix[1, 2] = -x_std * x_mean[1]
    n_matrix[2, 2] = 1
    return n_matrix @ x, n_matrix


def normalizer_Hartley_non_isotropic(x):
    x_2 = x @ x.T
    k = np.linalg.cholesky(x_2)
    n_matrix = np.linalg.inv(k)
    return n_matrix @ x, n_matrix


def normalizer_3dv2020(x, s, k):
    x_mean = np.mean(x, axis=1)
    t = np.array([[s, 0, -s * x_mean[0]], [0, s, -s * x_mean[1]], [0, 0, k ** abs(1 - x_mean[2])]])
    x_norm = np.dot(t, x)
    return x_norm, t


def mask_bearings_by_pm(all=False, **kwargs):
    if not all:
        delta = np.linalg.norm(
            (kwargs["bearings"]["kf"] - kwargs["bearings"]["frm"]), axis=0)
        max_value = np.max(delta)
        mask = delta > 0.8 * max_value
        if np.sum(mask) < 8:
            mask = delta > 0.5 * max_value
        else:
            mask = delta < max_value
    else:
        delta = np.linalg.norm(
            (kwargs["bearings"]["kf"] - kwargs["bearings"]["frm"]), axis=0)
        mask = delta < np.max(delta) + 1
    return mask


def normalizer_rsk(x, s, k, r):
    rot = eulerAnglesToRotationMatrix(r)[0:3, 0:3]
    n_matrix = np.eye(3) * s
    n_matrix[2, 2] = k
    t = rot @ n_matrix
    return t @ x, t
