from geometry_utilities import *
import cv2
from pcl_utilities import *
from scipy.optimize import least_squares


def Normalization(nd, x):
    '''
    Normalization of coordinates (centroid to the origin and mean distance of sqrt(2 or 3).

    Input
    -----
    nd: number of dimensions, 3 here
    x: the data to be normalized (directions at different columns and points at rows)
    Output
    ------
    Tr: the transformation matrix (translation plus scaling)
    x: the transformed data
    '''

    x = np.asarray(x)
    m, s = np.mean(x, 0), np.std(x)
    if nd == 2:
        Tr = np.array([[s, 0, m[0]], [0, s, m[1]], [0, 0, 1]])
    else:
        Tr = np.array([[s, 0, 0, m[0]], [0, s, 0, m[1]], [0, 0, s, m[2]],
                       [0, 0, 0, 1]])

    Tr = np.linalg.inv(Tr)
    x = np.dot(Tr, np.concatenate((x.T, np.ones((1, x.shape[0])))))
    x = x[0:nd, :].T

    return Tr, x


class PnP:
    """
    This class wraps the main functions for PnP problem.
    This implementation aims to find the p matrix to solve the PnP.
    """
    def __init__(self):
        self.params = None

    def build_matrix_A(self, x, w):
        """
        w: 3D point in camera world coordinates
        w = [u, v, w]
        """
        def compute_matrix_A_i(x, w, i):
            """
            A[i] = x_i_skew • w_i
            See https://www-users.cs.york.ac.uk/~wsmith/papers/TIP_SFM.pdf for more information
            """

            x_i_skew = vector2skew_matrix(x[:, i])
            w_i = np.array([
                # ! w_i matrix definition
                [w[0, i], w[1, i], w[2, i], 1, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, w[0, i], w[1, i], w[2, i], 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, w[0, i], w[1, i], w[2, i], 1]
            ])
            return x_i_skew @ w_i

            # '''
            # x, y, z = x[0, i], x[1, i], x[2, i]
            # u, v, w = w[0, i], w[1, i], w[2, i]
            #
            # A = []
            # A.append([x, y, z, 1, 0, 0, 0, 0, 0, 0, 0, 0, -u * x, -u * y, -u * z, -u])
            # A.append([0, 0, 0, 0, x, y, z, 1, 0, 0, 0, 0, -v * x, -v * y, -v * z, -v])
            # A.append([0, 0, 0, 0, 0, 0, 0, 0, x, y, z, 1, -w * x, -w * y, -w * z, -w])
            #
            # return np.array(A)
            # '''

        for i in range(x.shape[1]):
            if i == 0:
                A = compute_matrix_A_i(x, w, 0)
            else:
                A_i = compute_matrix_A_i(x, w, i)
                A = np.vstack((A, A_i))

        return A

    @staticmethod
    def compute_matrix_b(A):
        U, S, V = np.linalg.svd(A)
        return V[-1]
        # return V[-1, :]

    def recoverPose(self, x, w):
        # ! MP = 0
        Txyz, xyzn = Normalization(3, x)
        Tuvw, uvwn = Normalization(3, w)

        A = self.build_matrix_A(x, w)
        b = self.compute_matrix_b(A)
        b = b.reshape(3, 4)

        # ! b = [ R | t ] --> Force R into SO(3)
        P = approxRotationMatrix(b)
        T = np.eye(4)
        T[0:3, :] = P
        T[0:3, 3] = np.linalg.inv(T[0:3, 0:3]) @ T[0:3, 3]
        return T
