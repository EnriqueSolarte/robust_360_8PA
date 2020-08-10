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
                      [0, 0, k]])
        # t = np.array([[s, 0, 0],
        #               [0, s, 0],
        #               [0, 0, k ** abs(1 - x_mean[2])]])
        x_norm = np.dot(t, x)
        return x_norm, t

    def auto_normalizer(self, x1, x2):
        from delta_bound import eval_bound_by_bearings, get_delta_bound_by_bearings

        assert x1.shape == x2.shape
        assert x1.shape[0] == 3

        s, k = 1, 1
        param = [s, k]
        threshold = 0.9
        lr_s = 0.1
        delta = [0]
        i = 0
        while True:
            x1_norm, T1 = self.normalizer(x1.copy(), s=param[0], k=param[1])
            x2_norm, T2 = self.normalizer(x2.copy(), s=param[0], k=param[1])

            bound = eval_bound_by_bearings(x1_norm, x2_norm)
            delta.append(get_delta_bound_by_bearings(x1_norm, x2_norm))

            # bound = eval_bound_by_bearings(x1_norm_, x2_norm_)
            # C = get_frobenius_norm(x1_norm_, x2_norm_)
            # pm = np.mean(angle_between_vectors_arrays(x1_norm_, x2_norm_))

            if bound == 0:
                break

            if bound[0] < 0:
                delta.pop()
                param[i] += lr_s
                lr_s *= 0.1
            elif delta[-1] < delta[-2]:
                break
            elif param[i] <= threshold:
                threshold = param[i] * 0.9
                i = (i + 1) % 2
            else:
                param[i] -= lr_s
            if 0.01 > bound[0] > 0:
                break
            if abs(delta[-1] - np.nanmean(delta)) < 1e-1:
                break

        return x1_norm, x2_norm, T1, T2

    def lsq_normalizer(self, x1, x2):
        from delta_bound import eval_bound_by_bearings, get_delta_bound_by_bearings, get_frobenius_norm

        assert x1.shape == x2.shape
        assert x1.shape[0] == 3

        def residuals(x):
            x1_norm_, _ = self.normalizer(x1.copy(), s=x[0], k=x[1])
            x2_norm_, _ = self.normalizer(x2.copy(), s=x[0], k=x[1])

            delta_ = get_delta_bound_by_bearings(x1_norm_, x2_norm_)
            # bound_ = eval_bound_by_bearings(x1_norm_, x2_norm_)
            # C = get_frobenius_norm(x1_norm_, x2_norm_)
            pm = np.mean(angle_between_vectors_arrays(x1_norm_, x2_norm_))
            # loss = bound_[0] + 1/pm + 1/C
            A_norm = self.building_matrix_A(x1=x1_norm_, x2=x2_norm_)
            _, s, _ = np.linalg.svd(A_norm)
            if delta_ == np.nan:
                return np.inf
            return 1 / s[-2], 1 / delta_, 1 / pm

        initial = [1, 1]
        lsq = least_squares(residuals, initial)
        # bounds=([-np.inf, ], np.inf)
        # method="lm")

        x1_norm, T1 = self.normalizer(x1.copy(), s=lsq.x[0], k=lsq.x[1])
        x2_norm, T2 = self.normalizer(x2.copy(), s=lsq.x[0], k=lsq.x[1])
        delta = get_delta_bound_by_bearings(x1_norm, x2_norm)
        bound = eval_bound_by_bearings(x1_norm, x2_norm)
        # print("s:      {}  k:       {}".format(lsq.x[0], lsq.x[1]))
        # print("d-norm: {}  LB-norm: {}".format(delta, bound))
        delta = get_delta_bound_by_bearings(x1, x2)
        bound = eval_bound_by_bearings(x1, x2)
        # print("d:      {}  LB:      {}".format(delta, bound))

        return x1_norm, x2_norm, T1, T2

    def compute_essential_normalized(self, x1, x2, k=0.79, s=0.95):
        """
        #! THIS IS FOR DEVELOPMENT PURPOSES ONLY
        Computes the fundamental matrix from corresponding points
        (x1,x2 3*n arrays) using the normalized 8 point algorithm.
        """

        assert x1.shape == x2.shape, f"Shapes do not match {x1.shape} != {x2.shape}"
        assert x1.shape[0] in [3, 4], f"PCL out of shape {x1.shape} != (3, n) or (4, n)"

        x1, x2, T1, T2 = self.get_normalized_bearings(x1, x2, k=k, s=s)

        # compute F with the normalized coordinates
        E = self.compute_essential_matrix(x1, x2)

        # reverse normalization
        E = np.dot(T1.T, np.dot(E, T2))
        return E

    def recover_pose_from_matches(self, x1, x2):
        """
        return the a relative camera pose by using TLS method (Higgins 1981)
        """
        x1_norm, x2_norm, T1, T2 = self.lsq_normalizer(x1, x2)

        e = self.compute_essential_matrix(x1_norm, x2_norm)
        e = np.dot(T1.T, np.dot(e, T2))

        return self.recoverPose(e, x1, x2)
