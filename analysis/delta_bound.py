from pcl_utilities import *
from geometry_utilities import *
from solvers.epipolar_constraint import EightPointAlgorithmGeneralGeometry as g8p


def get_frobenius_norm(x1, x2):
    assert x1.shape == x2.shape
    assert x1.shape[0] == 3

    A = g8p.building_matrix_A(x1=x1, x2=x2)
    c_fro_norm = np.linalg.norm(A.T.dot(A), ord="fro")
    return c_fro_norm


def get_delta_bound_by_bearings(x1, x2):
    assert x1.shape == x2.shape
    assert x1.shape[0] == 3

    n = x1.shape[1]
    c_fro_norm = get_frobenius_norm(x1, x2)**2
    # return c_fro_norm
    # print(c_fro_norm)
    sqr_a = (8 * c_fro_norm - n**2) / 7

    sqr_b = (n / 8) - (1 / 8) * np.sqrt(sqr_a)
    delta = np.sqrt(sqr_b)
    return delta, c_fro_norm


def get_delta_bound(observed_matrix):
    """
    Returns the evaluation of the delta - gap value, which
    relates sensibility of a 8PA solution given the observed
    matrix A (n,9). n>8 matched points.
    """
    assert observed_matrix.shape[1] == 9

    # ! Equ (21) [Silveira CVPR 19']

    n = observed_matrix.shape[0]
    # ! c_fro_norm has  to be small
    c_fro_norm = np.linalg.norm(observed_matrix.T.dot(observed_matrix),
                                ord="fro")**2
    # print(c_fro_norm)
    sqr_a = (8 * c_fro_norm - n**2) / 7
    sqr_b = (n / 8) - (1 / 8) * np.sqrt(sqr_a)
    delta_bound = np.sqrt(sqr_b)
    return delta_bound
