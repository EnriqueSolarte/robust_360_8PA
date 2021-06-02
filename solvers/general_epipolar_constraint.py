from config import Cfg
import numpy as np
from utils import *


class EightPointAlgorithmGeneralGeometry:
    """
    This class wraps the main functions for the general epipolar geometry solution
    using bearing points which don't necessary lie on a homegeneous plane.
    This implementation aims to find the Essential matrix for both perspective and spherical projection.
    """

    # def __init__(self, cfg: Cfg):
    #     self.cfg = cfg

    def compute_essential_matrix(self, x1, x2, return_sigma=False):
        """
        This function compute the Essential matrix of a pair of Nx3 points. The points must be matched each other
        from two geometry views (Epipolar constraint). This general function doesn't assume a homogeneous
        representation of points.
        :param x1: Points from the 1st frame (n, 3) [x, y, z]
        :param x2: Points from the 2st frame (n, 3) [x, y, z]
        :return: Essential Matrix (3,3)
        """

        assert x1.shape == x2.shape, f"Shapes do not match {x1.shape} != {x2.shape}"
        assert x1.shape[0] in [3, 4], f"PCL out of shape {x1.shape} != (3, n) or (4, n)"

        A = self.building_matrix_A(x1, x2)

        #! compute linear least square solution
        U, Sigma, V = np.linalg.svd(A)
        E = V[-1].reshape(3, 3)

        #! constraint E
        #! making E rank 2 by setting out the last singular value
        U, S, V = np.linalg.svd(E)
        S[2] = 0
        E = np.dot(U, np.dot(np.diag(S), V))

        if return_sigma:
            return E / np.linalg.norm(E), Sigma
        return E / np.linalg.norm(E)

    @staticmethod
    def building_matrix_A(x1, x2):
        """
        Build an observation matrix A of the linear equation AX=0. This function doesn't assume
        homogeneous coordinates on a plane for p1s, and p2s
        :param x1:  Points from the 1st frame (n, 3) [x, y, z]
        :param x2: Points from the 2st frame (n, 3) [x, y, z]
        :return:  Matrix (n x 9)
        """
        A = np.array([
            x1[0, :] * x2[0, :], x1[0, :] * x2[1, :], x1[0, :] * x2[2, :],
            x1[1, :] * x2[0, :], x1[1, :] * x2[1, :], x1[1, :] * x2[2, :],
            x1[2, :] * x2[0, :], x1[2, :] * x2[1, :], x1[2, :] * x2[2, :]
        ]).T

        return A

    @staticmethod
    def get_the_four_cam_solutions_from_e(E, x1, x2):
        """
        This function computes the four relative transformation poses T(4, 4) 
        from x1 to x2 given an Essential matrix (3,3)
        :param E: Essential matrix
        :param x1: points in camera 1 (3, n) or (4, n)
        :param x2: points in camera 2 (3, n) or (4, n)
        """
        assert x1.shape == x2.shape
        assert x1.shape[0] == 3

        U, S, V = np.linalg.svd(E)
        if np.linalg.det(np.dot(U, V)) < 0:
            V = -V

        #! create matrix W and Z (Hartley's Book pp 258)
        W = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])

        t = U[:, 2].reshape(1, -1).T

        #! 4 possible transformations
        transformations = [
            np.vstack((np.hstack((U @ W.T @ V, t)), [0, 0, 0, 1])),
            np.vstack((np.hstack((U @ W.T @ V, -t)), [0, 0, 0, 1])),
            np.vstack((np.hstack((U @ W @ V, t)), [0, 0, 0, 1])),
            np.vstack((np.hstack((U @ W @ V, -t)), [0, 0, 0, 1])),
        ]
        return transformations

    def recover_pose_from_e(self, E, x1, x2):
        transformations = self.get_the_four_cam_solutions_from_e(E, x1, x2)
        return self.select_camera_pose(transformations, x1=x1, x2=x2)

    def select_camera_pose(self, transformations, x1, x2):
        """
        Selects the best transformation between a list of four posible tranformation given an 
        essential matrix. 
        :param transformations: list of (4x4) transformations
        :param x1: points in camera 1 (3, n) or (4, n)
        :param x2: points in camera 2 (3, n) or (4, n)
        """
        residuals = np.zeros((len(transformations),))
        # sample = np.random.randint(0, x1.shape[1],  int(x1.shape[1]*0.8))
        # x1 = np.copy(x1[:, sample])
        # x2 = np.copy(x2[:, sample])
        for idx, M in enumerate(transformations):
            pt1_3d = self.triangulate_points_from_cam_pose(cam_pose=M,
                                                           x1=x1,
                                                           x2=x2)
            # pt2_3d = np.linalg.inv(M).dot(pt1_3d)
            pt2_3d = M @ pt1_3d

            x1_hat = spherical_normalization(pt1_3d)
            x2_hat = spherical_normalization(pt2_3d)

            # ! Dot product xn and xn_hat must be close to 1
            closest_projections_cam2 = (np.sum(x2 * x2_hat, axis=0)) > 0.98
            closest_projections_cam1 = (np.sum(x1 * x1_hat, axis=0)) > 0.98
            residuals[idx] = np.sum(closest_projections_cam1) + np.sum(
                closest_projections_cam2)

        return transformations[residuals.argmax()]

    @staticmethod
    def get_e_from_cam_pose(cam_pose):

        t_x = vector2skew_matrix(cam_pose[0:3, 3] /
                                 np.linalg.norm(cam_pose[0:3, 3]))
        e = t_x.dot(cam_pose[0:3, 0:3])
        return e / np.linalg.norm(e)

    def recover_pose_from_matches(self, x1, x2):
        """
        Returns the a relative camera pose by using LSQ method (Higgins 1981)
        """
        e = self.compute_essential_matrix(x1, x2)
        return self.recover_pose_from_e(e, x1, x2)

    @staticmethod
    def triangulate_points_from_cam_pose(cam_pose, x1, x2):
        '''
        Triagulate 4D-points based on the relative camera pose and pts1 & pts2 matches
        :param Mn: Relative pose (4, 4) from cam1 to cam2
        :param x1: (3, n)
        :param x2: (3, n)
        :return:
        '''

        assert x1.shape[0] == 3
        assert x1.shape == x2.shape

        cam_pose = np.linalg.inv(cam_pose)
        landmarks_x1 = []
        for p1, p2 in zip(x1.T, x2.T):
            p1x = vector2skew_matrix(p1.ravel())
            p2x = vector2skew_matrix(p2.ravel())

            A = np.vstack(
                (np.dot(p1x,
                        np.eye(4)[0:3, :]), np.dot(p2x, cam_pose[0:3, :])))
            U, D, V = np.linalg.svd(A)
            landmarks_x1.append(V[-1])

        landmarks_x1 = np.asarray(landmarks_x1).T
        landmarks_x1 = landmarks_x1 / landmarks_x1[3, :]
        return landmarks_x1

    @staticmethod
    def projected_error(**kwargs):
        """
        This residual loss is introduced as projected distance in:
        A. Pagani and D. Stricker,
        “Structure from Motion using full spherical panoramic cameras,”
        ICCV 2011
        """
        E_dot_x1 = np.matmul(kwargs["e"].T, kwargs["x1"])
        E_dot_x2 = np.matmul(kwargs["e"], kwargs["x2"])
        dst = np.sum(kwargs["x1"] * E_dot_x2, axis=0)
        return dst / (np.linalg.norm(kwargs["x1"]) *
                    np.linalg.norm(E_dot_x2))

    @staticmethod
    def algebraic_error(**kwargs):
        E_dot_x2 = np.matmul(kwargs["e"], kwargs["x2"])
        dst = np.sum(kwargs["x1"] * E_dot_x2, axis=0)
        return dst