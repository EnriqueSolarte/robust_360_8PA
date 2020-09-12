from solvers.epipolar_constraint import EightPointAlgorithmGeneralGeometry
from geometry_utilities import *
from scipy.optimize import least_squares
import levmar


class Optimal8PA(EightPointAlgorithmGeneralGeometry):
    """
    This Class is the VSLAB implementation of the norm 8PA with optimal K,S
    for spherical projection models
    """

    def __init__(self, version='v1'):
        super().__init__()
        self.T1 = np.eye(3)
        self.T2 = np.eye(3)
        self.version = version
        if version == 'v0':
            """
            Version submitted at 3DV 2020
            """
            self.optimizer_function = self.optimizer_v0
        if version == 'v1':
            """ 
            This version is currently the one which is working only 
            in synthetic points/features
            """
            self.optimizer_function = self.optimizer_v1
        if version == "v1.1":
            """
            This version optimizes E based on the residual function
            """
            self.optimizer_function = self.optimizer_v1_1

        if version == "v1.0.1":
            """
            This version uses a set of triangulated points to evaluate
            a reprojection error, therefore optimizes K, S.
            This version considers that the triangulated points
            are constant all the time.
            """
            self.optimizer_function = self.optimizer_v1_0_1
            self.landmarks_kf = None

        if version == "vR.t":
            """
            This version implement the PnP iterative. It uses
            a set of landmarks and its corresponded projection as 
            bearing vectors. 
            """
            self.landmarks_kf = None

        if version == 'v2':
            self.optimizer_function = self.optimizer_v2
        if version == 'v2.1':
            self.optimizer_function = self.optimizer_v2_1

    @staticmethod
    def normalizer(x, s, k):
        # x_mean = np.mean(x, axis=1)
        # rot = get_rot_from_directional_vectors(x_mean, (0, 0, 1))
        # x_rot = rot.dot(x)
        # t = np.array([[s, 0, k],
        #               [0, s, k],
        #               [0, 0, k]])
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
        from analysis.utilities.stability_utilities import get_frobenius_norm

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
        from analysis.utilities.stability_utilities import get_delta_bound_by_bearings

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
        from analysis.utilities.stability_utilities import get_frobenius_norm

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

    def optimizer_v1_0_1(self, x1, x2):

        def reprojection_error(parameters, x1, x2):
            s1, k1 = parameters[0], parameters[1]
            s2, k2 = parameters[2], parameters[3]

            x1_norm_, t1 = self.normalizer(x1.copy(), s=s1, k=k1)
            x2_norm_, t2 = self.normalizer(x2.copy(), s=s2, k=k2)

            e_norm = self.compute_essential_matrix(
                x1=x1_norm_,
                x2=x2_norm_
            )
            e_hat = t1.T @ e_norm @ t2
            self.cam_hat = self.recover_pose_from_e(
                E=e_hat,
                x1=x1,
                x2=x2
            )
            # if self.landmarks_kf is not None:
            self.landmarks_kf = self.triangulate_points_from_cam_pose(
                cam_pose=self.cam_hat,
                x1=x1,
                x2=x2
            )

            # sample = np.random.randint(0, self.landmarks_kf.shape[1], 8)
            landmarks_frm_hat = np.linalg.inv(self.cam_hat) @ self.landmarks_kf
            error = get_angle_between_vectors_arrays(
                array_ref=x2,
                array_vector=landmarks_frm_hat[0:3, :]
            )
            return error

        initial_parameters = np.array((1, 1, 1, 1))
        self.landmarks_kf = None
        opt_k_s, p_cov, info = levmar.levmar(
            reprojection_error,
            initial_parameters,
            np.ones_like(x1[0, :]),
            args=(x1.copy(),
                  x2.copy()))

        s1, k1 = opt_k_s[0], opt_k_s[1]

        s2, k2 = opt_k_s[2], opt_k_s[3]

        return s1, s2, k1, k2

    def optimum_normalizer(self, x1, x2):
        s1, s2, k1, k2 = self.optimizer_function(x1, x2)

        x1_norm, T1 = self.normalizer(x1.copy(), s=s1, k=k1)
        x2_norm, T2 = self.normalizer(x2.copy(), s=s2, k=k2)

        print("S1   :{0:0.3f} - K1        :{1:0.3f}".format(s1, k1))
        print("S2   :{0:0.3f} - K2        :{1:0.3f}".format(s2, k2))

        return x1_norm, x2_norm, T1, T2

    def recover_pose_from_matches(self, x1, x2, param=None, eval_current_solution=False):
        """
        return the a relative camera pose by using TLS method (Higgins 1981)
        """
        assert x1.shape == x2.shape
        assert x1.shape[0] == 3

        if param is None:
            x1_norm, x2_norm, self.T1, self.T2 = self.optimum_normalizer(x1, x2)
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

    def recover_pose_and_optimize(self, x1, x2, eval_current_solution=False):
        """
        return the a relative camera pose by using TLS method (Higgins 1981)
        """
        assert x1.shape[1] == x2.shape[1]

        e = self.compute_essential_matrix(x1, x2)

        cam_hat = self.recover_pose_from_e(e, x1, x2)
        self.landmarks_kf = self.triangulate_points_from_cam_pose(
            cam_pose=cam_hat,
            x1=x1,
            x2=x2
        )

        eu = rotationMatrixToEulerAngles(cam_hat[0:3, 0:3])
        trn = np.copy(cam_hat[0:3, 3])
        initial_R_t = np.hstack((eu, trn))

        def reprojection_error(parameters, _landmarks_kf, _bearings_frm):
            r0 = parameters[0]
            r1 = parameters[1]
            r2 = parameters[2]
            t0 = parameters[3]
            t1 = parameters[4]
            t2 = parameters[5]

            cam_pose = eulerAnglesToRotationMatrix((r0, r1, r2))
            cam_pose[0:3, 3] = np.array((t0, t1, t2))

            landmarks_frm_hat = np.linalg.inv(cam_pose) @ _landmarks_kf
            error = get_angle_between_vectors_arrays(
                array_ref=_bearings_frm,
                array_vector=landmarks_frm_hat[0:3, :]
            )
            return error

        opt_R_t, p_cov, info = levmar.levmar(
            reprojection_error,
            initial_R_t,
            np.zeros_like(x2[0, :]),
            args=(self.landmarks_kf, x2))

        cam_final = eulerAnglesToRotationMatrix(opt_R_t[0:3])
        cam_final[0:3, 3] = opt_R_t[3:]

        if eval_current_solution:
            e_hat = self.get_e_from_cam_pose(cam_final)
            self.current_count_features = x1.shape[1]
            self.current_residual = np.sum(self.residual_function_evaluation(
                e=e_hat,
                x1=x1,
                x2=x2
            ) ** 2)

        return cam_final

    # * Methods needed for RANSAC
    def estimate(self, *data):
        x1, x2 = data[0].T, data[1].T
        x1_norm, x2_norm, self.T1, self.T2 = self.optimum_normalizer(x1, x2)
        e = self.compute_essential_matrix(x1_norm, x2_norm)
        self.params = np.dot(self.T1.T, np.dot(e, self.T2))
        return True
