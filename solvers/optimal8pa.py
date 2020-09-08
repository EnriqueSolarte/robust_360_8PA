from solvers.epipolar_constraint import EightPointAlgorithmGeneralGeometry
from geometry_utilities import *
from scipy.optimize import least_squares
from solvers.epipolar_constraint import projected_distance, sampson_distance, tangential_distance
from utilities.stability_utilities import get_frobenius_norm


class Optimal8PA(EightPointAlgorithmGeneralGeometry):
    """
    This Class is the VSLAB implementation of the optimal 8PA
    for perspective and spherical projection models
    """

    def __init__(self, version='v1', residual_function=None):
        super().__init__()
        if residual_function is not None:
            self.residual_function_evaluation = residual_function
        self.T1 = np.eye(3)
        self.T2 = np.eye(3)
        self.version = version
        if version == 'v0':
            """
            Version submitted at 3DV 2020
            """
            self.optimize_parameters = self.optimizer_v0
        if version == 'v1':
            """ 
            This version is curretnly the one which is working only 
            in synthetic points/features
            """
            self.optimize_parameters = self.optimizer_v1
        if version == "v1.1":
            """
            This version optimize E based on the residual function
            """
            self.optimize_parameters = self.optimizer_v1_1

        if version == "v1.2":
            """
            This version optimize E based on the residual function but with TWO normalizer matrices
            """
            self.optimize_parameters = self.optimizer_v1_2

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
        from utilities.stability_utilities import get_frobenius_norm

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

    def optimizer_v1_1(self, x1, x2):
        def residuals(x):
            x1_norm_, t1 = self.normalizer(x1.copy(), s=x[0], k=x[1])
            x2_norm_, t2 = self.normalizer(x2.copy(), s=x[0], k=x[1])
            e_norm = self.compute_essential_matrix(x1=x1_norm_, x2=x2_norm_)
            e = np.dot(t1, np.dot(e_norm, t2))
            # C, A = get_frobenius_norm(x1_norm_, x2_norm_, return_A=True)
            # _, sigma, _ = np.linalg.svd(A)
            return np.sum(self.residual_function_evaluation(e=e, x1=x1, x2=x2) ** 2)

        initial = [1, 1]
        lsq = least_squares(residuals, initial)
        s = lsq.x[0]
        k = lsq.x[1]
        return s, s, k, k

    def optimizer_v1_2(self, x1, x2):
        def residuals(x):
            x1_norm_, t1 = self.normalizer(x1.copy(), s=x[0], k=x[1])
            x2_norm_, t2 = self.normalizer(x2.copy(), s=x[2], k=x[3])
            e_norm = self.compute_essential_matrix(x1=x1_norm_, x2=x2_norm_)
            e = np.dot(t1, np.dot(e_norm, t2))
            # C, A = get_frobenius_norm(x1_norm_, x2_norm_, return_A=True)
            # _, sigma, _ = np.linalg.svd(A)
            return np.sum(self.residual_function_evaluation(e=e, x1=x1, x2=x2) ** 2)

        initial = [1, 0.1, 0.1, 0.1]
        lsq = least_squares(residuals, initial, method="lm")
        s1 = lsq.x[0]
        k1 = lsq.x[1]
        s2 = lsq.x[2]
        k2 = lsq.x[3]

        return s1, s2, k1, k2

    def optimizer_v2(self, x1, x2):
        from utilities.stability_utilities import get_delta_bound_by_bearings

        def residuals(x):
            x1_norm_, _ = self.normalizer(x1.copy(), s=x[0], k=x[1])
            x2_norm_, _ = self.normalizer(x2.copy(), s=x[2], k=x[3])

            delta_, C = get_delta_bound_by_bearings(x1_norm_, x2_norm_)
            pm = np.degrees(
                np.nanmean(get_angle_between_vectors_arrays(x1_norm_, x2_norm_)))
            if delta_ == np.nan:
                return np.inf
            return C / delta_

        initial = [1, 1, 1, 1]
        lsq = least_squares(residuals, initial)
        s1, k1 = lsq.x[0], lsq.x[1]
        s2, k2 = lsq.x[2], lsq.x[3]
        return s1, s2, k1, k2

    def optimizer_v2_1(self, x1, x2):
        from utilities.stability_utilities import get_frobenius_norm

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
        from utilities.stability_utilities import get_delta_bound_by_bearings

        s1, s2, k1, k2 = self.optimize_parameters(x1, x2)

        x1_norm, T1 = self.normalizer(x1.copy(), s=s1, k=k1)
        x2_norm, T2 = self.normalizer(x2.copy(), s=s2, k=k2)
        # delta_norm, C_norm2 = get_delta_bound_by_bearings(x1_norm, x2_norm)
        # delta, C2 = get_delta_bound_by_bearings(x1, x2)
        # pm_norm = np.nanmean(angle_between_vectors_arrays(x1_norm, x2_norm))
        # pm = np.nanmean(angle_between_vectors_arrays(x1, x2))

        print("S1   :{0:0.3f} - K1        :{1:0.3f}".format(s1, k1))
        print("S2   :{0:0.3f} - K2        :{1:0.3f}".format(s2, k2))
        # print("delta:{0:0.3f} - delta_norm:{1:0.3f}".format(delta, delta_norm))
        # print("C2   :{0:0.3f} - C_norm2   :{1:0.5f}".format(C2, C_norm2))
        # print("pm   :{0:0.3f} - pm_norm   :{1:0.5f}".format(pm, pm_norm))

        return x1_norm, x2_norm, T1, T2

    def recover_pose_from_matches(self, x1, x2, param=None, eval_current_solution=False):
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

        if eval_current_solution:
            self.current_count_features = x1.shape[1]
            self.current_residual = np.sum(self.residual_function_evaluation(
                e=e,
                x1=x1,
                x2=x2
            ) ** 2)

        return self.recover_pose_from_e(e, x1, x2)

    # * Methods needed for RANSAC
    def estimate(self, *data):
        x1, x2 = data[0].T, data[1].T
        x1_norm, x2_norm, self.T1, self.T2 = self.lsq_normalizer(x1, x2)
        e = self.compute_essential_matrix(x1_norm, x2_norm)
        self.params = np.dot(self.T1.T, np.dot(e, self.T2))
        return True
