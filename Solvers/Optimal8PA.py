from Solvers.EpipolarConstraint import EightPointAlgorithmGeneralGeometry
from geometry_utilities import *
from scipy.optimize import least_squares


class Optimal8PA(EightPointAlgorithmGeneralGeometry):
    """
    This Class is the VSLAB implementation of the optimal 8PA
    for perspective and spherical projection models
    """

    def __init__(self):
        super().__init__()

    @staticmethod
    def normalizer(x, s, k):
        # x_mean = np.mean(x, axis=1)
        # rot = get_rot_from_directional_vectors(x_mean, (0, 0, 1))
        # x_rot = rot.dot(x)
        # t = np.array([[s, 0, -s * x_mean[0]],
        #               [0, s, -s * x_mean[1]],
        #               [0, 0, k ** abs(1 - x_mean[2])]])
        t = np.array([[s, 0, 0],
                      [0, s, 0],
                      [0, 0, 1]])
        # t = np.array([[s, 0, 0],
        #               [0, s, 0],
        #               [0, 0, k ** abs(1 - x_mean[2])]])
        x_norm = np.dot(t, x)
        return x_norm, t

    def lsq_normalizer(self, x1, x2):
        from delta_bound import get_delta_bound_by_bearings
        assert x1.shape == x2.shape
        assert x1.shape[0] == 3

        def residuals(x):
            x1_norm_, _ = self.normalizer(x1.copy(), s=x[0], k=x[1])
            x2_norm_, _ = self.normalizer(x2.copy(), s=x[0], k=x[1])

            delta_, C = get_delta_bound_by_bearings(x1_norm_, x2_norm_)
            pm = np.mean(angle_between_vectors_arrays(x1_norm_, x2_norm_))
            if delta_ == np.nan:
                return np.inf
            return C / delta_

        initial = [1, 1]
        lsq = least_squares(residuals, initial)
        x1_norm, T1 = self.normalizer(x1.copy(), s=lsq.x[0], k=lsq.x[1])
        x2_norm, T2 = self.normalizer(x2.copy(), s=lsq.x[0], k=lsq.x[1])
        delta_norm, C_norm2 = get_delta_bound_by_bearings(x1_norm, x2_norm)
        delta, C2 = get_delta_bound_by_bearings(x1, x2)
        print("S    :{0:0.3f} - K         :{1:0.3f}".format(lsq.x[0], lsq.x[1]))
        print("delta:{0:0.3f} - delta_norm:{1:0.3f}".format(delta, delta_norm))
        print("C2   :{0:0.3f} - C_norm2   :{1:0.5f}".format(C2, C_norm2))

        return x1_norm, x2_norm, T1, T2

    def recover_pose_from_matches(self, x1, x2):
        """
        return the a relative camera pose by using TLS method (Higgins 1981)
        """
        x1_norm, x2_norm, T1, T2 = self.lsq_normalizer(x1, x2)

        e = self.compute_essential_matrix(x1_norm, x2_norm)
        e = np.dot(T1.T, np.dot(e, T2))

        return self.recoverPose(e, x1, x2)
