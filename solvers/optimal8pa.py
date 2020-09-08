from solvers.epipolar_constraint import EightPointAlgorithmGeneralGeometry
from geometry_utilities import *
from scipy.optimize import least_squares


class Optimal8PA(EightPointAlgorithmGeneralGeometry):
    """
    This Class is the VSLAB implementation of the optimal 8PA
    for perspective and spherical projection models
    """
    def __init__(self, version='v1'):
        super().__init__()
        self.T1 = np.eye(3)
        self.T2 = np.eye(3)
        self.version = version
        if version == 'v0':
            self.optimize_parameters = self.optimizer_v0
        if version == 'v1':
            self.optimize_parameters = self.optimizer_v1
        if version == 'v2':
            self.optimize_parameters = self.optimizer_v2
        if version == 'v2.1':
            self.optimize_parameters = self.optimizer_v2_1

    @staticmethod
    def normalizer(x, s, k):
        # x_mean = np.mean(x, axis=1)
        # rot = get_rot_from_directional_vectors(x_mean, (0, 0, 1))
        # x_rot = rot.dot(x)
        # t = np.array([[s, 0, -s * x_mean[0]],
        #               [0, s, -s * x_mean[1]],
        #               [0, 0, k ** abs(1 - x_mean[2])]])
        t = np.array([[s, 0, 0], [0, s, 0], [0, 0, k]])
        # t = np.array([[s, 0, 0],
        #               [0, s, 0],
        #               [0, 0, k ** abs(1 - x_mean[2])]])
        x_norm = np.dot(t, x)
        return x_norm, t

    @staticmethod
    def loss(C, delta, pm):
        return C / delta

    def optimizer_v0(self, x1, x2):
        s = 2
        k = 10
        return s, s, k, k

    def optimizer_v1(self, x1, x2):
        from analysis.delta_bound import get_frobenius_norm

        def residuals(x):
            x1_norm_, _ = self.normalizer(x1.copy(), s=x[0], k=x[1])
            x2_norm_, _ = self.normalizer(x2.copy(), s=x[0], k=x[1])

            C, A = get_frobenius_norm(x1_norm_, x2_norm_, return_A=True)
            _, sigma, _ = np.linalg.svd(A)
            # pm = np.degrees(
            #     np.nanmean(angle_between_vectors_arrays(x1_norm_, x2_norm_)))
            return self.loss(C, sigma[-2], None)

        initial = [1, 1]
        lsq = least_squares(residuals, initial)
        s = lsq.x[0]
        k = lsq.x[1]
        return s, s, k, k

    def optimizer_v2(self, x1, x2):
        from analysis.delta_bound import get_delta_bound_by_bearings

        def residuals(x):
            x1_norm_, _ = self.normalizer(x1.copy(), s=x[0], k=x[1])
            x2_norm_, _ = self.normalizer(x2.copy(), s=x[2], k=x[3])

            delta_, C = get_delta_bound_by_bearings(x1_norm_, x2_norm_)
            pm = np.degrees(
                np.nanmean(angle_between_vectors_arrays(x1_norm_, x2_norm_)))
            if delta_ == np.nan:
                return np.inf
            return C / delta_

        initial = [1, 1, 1, 1]
        lsq = least_squares(residuals, initial)
        s1, k1 = lsq.x[0], lsq.x[1]
        s2, k2 = lsq.x[2], lsq.x[3]
        return s1, s2, k1, k2

    def optimizer_v2_1(self, x1, x2):
        from analysis.delta_bound import get_frobenius_norm

        def residuals(x):
            x1_norm_, _ = self.normalizer(x1.copy(), s=x[0], k=x[1])
            x2_norm_, _ = self.normalizer(x2.copy(), s=x[2], k=x[3])

            C, A = get_frobenius_norm(x1_norm_, x2_norm_, return_A=True)
            _, sigma, _ = np.linalg.svd(A)
            # pm = np.degrees(
            #     np.nanmean(angle_between_vectors_arrays(x1_norm_, x2_norm_)))
            return C / sigma[-2]

        initial = [1, 1, 1, 1]
        lsq = least_squares(residuals, initial)
        s1, k1 = lsq.x[0], lsq.x[1]
        s2, k2 = lsq.x[2], lsq.x[3]
        return s1, s2, k1, k2

    def lsq_normalizer(self, x1, x2):
        assert x1.shape == x2.shape
        assert x1.shape[0] == 3
        from analysis.delta_bound import get_delta_bound_by_bearings

        s1, s2, k1, k2 = self.optimize_parameters(x1, x2)

        x1_norm, T1 = self.normalizer(x1.copy(), s=s1, k=k1)
        x2_norm, T2 = self.normalizer(x2.copy(), s=s2, k=k2)
        # delta_norm, C_norm2 = get_delta_bound_by_bearings(x1_norm, x2_norm)
        # delta, C2 = get_delta_bound_by_bearings(x1, x2)
        # pm_norm = np.nanmean(angle_between_vectors_arrays(x1_norm, x2_norm))
        # pm = np.nanmean(angle_between_vectors_arrays(x1, x2))

        # print("S1   :{0:0.3f} - K1        :{1:0.3f}".format(s1, k1))
        # print("S2   :{0:0.3f} - K2        :{1:0.3f}".format(s2, k2))
        # print("delta:{0:0.3f} - delta_norm:{1:0.3f}".format(delta, delta_norm))
        # print("C2   :{0:0.3f} - C_norm2   :{1:0.5f}".format(C2, C_norm2))
        # print("pm   :{0:0.3f} - pm_norm   :{1:0.5f}".format(pm, pm_norm))

        return x1_norm, x2_norm, T1, T2

    def recover_pose_from_matches(self, x1, x2, param=None):
        """
        return the a relative camera pose by using TLS method (Higgins 1981)
        """
        if param is None:
            x1_norm, x2_norm, self.T1, self.T2 = self.lsq_normalizer(x1, x2)
        else:
            x1_norm, self.T1, = self.normalizer(x1,
                                                s=param[0][0],
                                                k=param[0][1])
            x2_norm, self.T2, = self.normalizer(x2,
                                                s=param[1][0],
                                                k=param[1][1])

        e = self.compute_essential_matrix(x1_norm, x2_norm)
        e = np.dot(self.T1.T, np.dot(e, self.T2))

        return self.recoverPose(e, x1, x2)

    def compute_essential_matrix(self, x1, x2):
        """
        This function compute the Essential matrix of a pair of Nx3 points. The points must be match each other
        from two geometry views (Epipolar constraint). This general function doesn't assume a homogeneous
        representation of points.
        :param x1: Points from the 1st frame (n, 3) [x, y, z]
        :param x2: Points from the 2st frame (n, 3) [x, y, z]
        :return: Essential Matrix (3,3)
        """
        assert x1.shape == x2.shape, f"Shapes do not match {x1.shape} != {x2.shape}"
        assert x1.shape[0] in [
            3, 4
        ], f"PCL out of shape {x1.shape} != (3, n) or (4, n)"

        A = self.building_matrix_A(x1, x2)
        # compute linear least square solution
        U, S, V = np.linalg.svd(A)
        E = V[-1].reshape(3, 3)

        # ! constraint E
        # make rank 2 by zeroing out last singular value
        U, S, V = np.linalg.svd(E)
        # S[0] = (S[0] + S[1]) / 2.0
        # S[1] = S[0]
        S[2] = 0
        E = np.dot(U, np.dot(np.diag(S), V))
        return E / np.linalg.norm(E)

    # * Methods needed for RANSAC
    def estimate(self, *data):
        x1, x2 = data[0].T, data[1].T
        x1_norm, x2_norm, self.T1, self.T2 = self.lsq_normalizer(x1, x2)
        e = self.compute_essential_matrix(x1_norm, x2_norm)
        self.params = np.dot(self.T1.T, np.dot(e, self.T2))
        return True
