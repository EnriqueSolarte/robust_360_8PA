import levmar
import numpy as np
from solvers.general_epipolar_constraint import EightPointAlgorithmGeneralGeometry as G8PA

g8p = G8PA()


def normalizer_s_k(x, s, k):
    n_matrix = np.eye(3) * s
    n_matrix[2, 2] = k
    return n_matrix @ x, n_matrix


def residuals_error_SK(parameters, bearings_kf, bearings_frm):
    s = parameters[0]
    k = parameters[1]
    bearings_kf_norm, t1 = normalizer_s_k(x=bearings_kf, s=s, k=k)
    bearings_frm_norm, t2 = normalizer_s_k(x=bearings_frm, s=s, k=k)

    e_norm = g8p.compute_essential_matrix(
        x1=bearings_kf_norm, x2=bearings_frm_norm)
    # # # ! De-normalization
    e_hat = t1.T @ e_norm @ t2
    residuals_error = g8p.projected_error(
        e=e_hat, x1=bearings_kf, x2=bearings_frm)

    # * KS: L1
    loss = np.zeros_like(parameters)
    loss[0] = np.sum(np.abs(residuals_error))

    return loss


def get_cam_pose_by_opt_SK(x1, x2):

    opt_k_s, p_cov, info = levmar.levmar(
        func=residuals_error_SK,
        p0=(1, 1),
        y=np.zeros((2,)),
        args=(x1.copy(),
              x2.copy()))

    s = opt_k_s[0]
    k = opt_k_s[1]

    bearings_kf_norm, t1 = normalizer_s_k(
        x=x1.copy(), s=s, k=k)
    bearings_frm_norm, t2 = normalizer_s_k(
        x=x2.copy(), s=s, k=k)

    e_norm = g8p.compute_essential_matrix(
        x1=bearings_kf_norm, x2=bearings_frm_norm)
    e_hat = t1.T @ e_norm @ t2

    cam_hat = g8p.recover_pose_from_e(
        E=e_hat,
        x1=x1.copy(),
        x2=x2.copy(),
    )

    return cam_hat


def get_cam_pose_by_8pa(x1, x2):
    """
    Returns a camera pose using bearing vectors and the 8PA
    """
    cam_pose = g8p.recover_pose_from_matches(
        x1=x1.copy(),
        x2=x2.copy(),
    )

    return cam_pose
