import levmar
import numpy as np
from solvers.general_epipolar_constraint import EightPointAlgorithmGeneralGeometry as G8PA
from liegroups.numpy import SE3
from scipy.stats import norm

g8p = G8PA()


def normalizer_s_k(x, s, k):
    n_matrix = np.eye(3) * s
    n_matrix[2, 2] = k
    return n_matrix @ x, n_matrix


def residuals_error_w(parameters, bearings_kf, bearings_frm, weights):
    cam_pose = SE3.exp(parameters).as_matrix()

    e_rt = g8p.get_e_from_cam_pose(cam_pose)

    residuals_error = g8p.projected_error(
        e=e_rt, x1=bearings_kf, x2=bearings_frm)

    loss = np.sqrt(weights) * residuals_error
    return loss


def residuals_error_SK(parameters, bearings_kf, bearings_frm):
    s = parameters[0]
    k = parameters[1]
    bearings_kf_norm, t1 = normalizer_s_k(x=bearings_kf, s=s, k=k)
    bearings_frm_norm, t2 = normalizer_s_k(x=bearings_frm, s=s, k=k)

    e_norm = g8p.compute_essential_matrix(
        x1=bearings_kf_norm, x2=bearings_frm_norm)
    
    ## ! De-normalization
    e_hat = t1.T @ e_norm @ t2
    residuals_error = g8p.projected_error(
        e=e_hat, x1=bearings_kf, x2=bearings_frm)

    loss = np.zeros_like(parameters)
    loss[0] = np.sum(np.abs(residuals_error))

    return loss


def get_cam_pose_by_opt_SK(x1, x2):
    """
    Returns a camera pose by optimazing K and S.
    This is the robust normalized 8PA for 360-images 
    """
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


def get_cam_pose_by_GSM_const_wSK(x1, x2):
    """
    Returns a camera pose by a constant weighted least-square optimization of
    residuals error (weithed GSM). The constant weights are computed by using residuals from the 
    robust normalized 8PA
    """
    opt_k_s, p_cov, info = levmar.levmar(
        func=residuals_error_SK,
        p0=(1, 1),
        y=np.zeros((2,)),
        args=(x1.copy(),
              x2.copy(),))

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

    residual = g8p.projected_error(
        e=e_hat,
        x1=x1.copy(),
        x2=x2.copy())
    weights = norm(np.mean(residual), np.std(residual))

    xi = SE3.log(SE3.from_matrix(cam_hat))
    opt_rt, p_cov, info = levmar.levmar(
        func=residuals_error_w,
        y=np.zeros_like(x1[0, :]),
        p0=xi,
        args=(x1.copy(),
              x2.copy(),
              weights.pdf(residual)
              ))
    cam_final = SE3.exp(opt_rt).as_matrix()

    return cam_final


def residuals_error_GSM(parameters, bearings_kf, bearings_frm):
    """
    Loss function for GSM(R,t)
    """
    cam_pose = SE3.exp(parameters).as_matrix()
    residual = g8p.projected_error(
        e=g8p.get_e_from_cam_pose(cam_pose),
        x1=bearings_kf,
        x2=bearings_frm)
    # * RT: L2
    loss = residual
    return loss


def get_cam_pose_by_8pa(x1, x2):
    """
    Returns a camera pose by 8PA
    """
    cam_pose = g8p.recover_pose_from_matches(
        x1=x1.copy(),
        x2=x2.copy(),
    )

    return cam_pose


def get_cam_pose_by_GSM(x1, x2):
    """
    Returns a camera pose by least-square optmization of the residual errors.
    """
    initial_pose = get_cam_pose_by_8pa(x1, x2)
    xi = SE3.log(SE3.from_matrix(initial_pose))
    opt_r_t, p_cov, info = levmar.levmar(
        func=residuals_error_GSM,
        p0=xi,
        y=np.zeros_like(x1[0, :]),
        args=(x1, x2))

    cam_final = SE3.exp(opt_r_t).as_matrix()
    return cam_final


def get_cam_pose_by_GSM_const_wRT(x1, x2):
    """
    Returns a camera pose by a constant weighted least-square optimization of
    residuals error (weithed GSM). The constant weights are computed by using residuals from the 8PA
    """
    cam_hat = get_cam_pose_by_8pa(x1, x2)
    residual = g8p.projected_error(
        e=g8p.get_e_from_cam_pose(cam_hat),
        x1=x1.copy(),
        x2=x2.copy())

    weights = norm(np.mean(residual), np.std(residual))

    xi = SE3.log(SE3.from_matrix(cam_hat))
    opt_rt, p_cov, info = levmar.levmar(
        func=residuals_error_w,
        y=np.zeros_like(x1[0, :]),
        p0=xi,
        args=(x1.copy(),
              x2.copy(),
              weights.pdf(residual)
              ))
    cam_final = SE3.exp(opt_rt).as_matrix()

    return cam_final
