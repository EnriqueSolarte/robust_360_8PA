from solvers.epipolar_constraint import EightPointAlgorithmGeneralGeometry as g8p
import levmar
from analysis.utilities.optmization_utilities import *
from solvers.epipolar_constraint import *
import time
from liegroups.numpy import SE3
from scipy.stats import norm
from scipy.stats import t
from analysis.utilities.camera_recovering import *

solver = g8p()


def get_ablation_data_by_8pa(**kwargs):
    """
    Returns a camera pose using bearing vectors and the 8PA
    """
    delta_time = time.time()
    e_hat, sigma, A = solver.compute_essential_matrix(
        x1=kwargs["bearings"]['kf'].copy(),
        x2=kwargs["bearings"]['frm'].copy(),
        return_all=True
    )
    time_evaluation = time.time() - delta_time
    delta_time = time.time()
    cam_pose = solver.recover_pose_from_e(
        E=e_hat,
        x1=kwargs["bearings"]['kf'].copy(),
        x2=kwargs["bearings"]['frm'].copy(),
    )
    time_evaluation = time.time() - delta_time + time_evaluation

    paralax_motion = get_angle_between_vectors_arrays(
        kwargs["bearings"]['kf'].copy(),
        kwargs["bearings"]['frm'].copy()
    )
    return cam_pose, time_evaluation, sigma, np.linalg.norm(A, ord="fro"), paralax_motion


# * ****************************************************************
# ! OURS - OPT SK (residuals)

def get_ablation_data_by_opt_SK(**kwargs):
    initial_s_k = kwargs[
        "iVal_Res_SK"]
    delta_time = time.time()
    opt_k_s, p_cov, info = levmar.levmar(
        func=residuals_error_SK,
        p0=initial_s_k,
        y=np.zeros_like(initial_s_k),
        args=(kwargs["bearings"]["kf"].copy(),
              kwargs["bearings"]["frm"].copy()))

    time_evaluation = time.time() - delta_time
    s = opt_k_s[0]
    k = opt_k_s[1]
    delta_time = time.time()
    bearings_kf_norm, t1 = normalizer_s_k(
        x=kwargs["bearings"]["kf"].copy(), s=s, k=k)
    bearings_frm_norm, t2 = normalizer_s_k(
        x=kwargs["bearings"]["frm"].copy(), s=s, k=k)

    e_norm, sigma, A = solver.compute_essential_matrix(
        x1=bearings_kf_norm, x2=bearings_frm_norm, return_all=True)

    e_hat = t1.T @ e_norm @ t2
    cam_hat = solver.recover_pose_from_e(
        E=e_hat,
        x1=kwargs["bearings"]['kf'].copy(),
        x2=kwargs["bearings"]['frm'].copy(),
    )
    time_evaluation += time.time() - delta_time

    paralax_motion = get_angle_between_vectors_arrays(
        bearings_kf_norm, bearings_frm_norm
    )

    return cam_hat, time_evaluation, sigma, np.linalg.norm(A, ord="fro"), paralax_motion


def get_ablation_data_by_Hartley_non_isotropic_isotropic(**kwargs):
    tic_toc = time.time()
    bearings_kf_norm, t1 = normalizer_Hartley_non_isotropic(
        x=kwargs["bearings"]["kf"].copy())
    bearings_frm_norm, t2 = normalizer_Hartley_isotropic(
        x=kwargs["bearings"]["frm"].copy())

    e_norm, sigma, A = solver.compute_essential_matrix(
        x1=bearings_kf_norm, x2=bearings_frm_norm, return_all=True)

    e_hat = t1.T @ e_norm @ t2
    cam_hat = solver.recover_pose_from_e(
        E=e_hat,
        x1=kwargs["bearings"]['kf'].copy(),
        x2=kwargs["bearings"]['frm'].copy(),
    )
    time_evaluation = time.time() - tic_toc

    paralax_motion = get_angle_between_vectors_arrays(
        bearings_kf_norm, bearings_frm_norm
    )

    return cam_hat, time_evaluation, sigma, np.linalg.norm(A, ord="fro"), paralax_motion


def get_ablation_data_by_Hartley_non_isotropic(**kwargs):
    tic_toc = time.time()
    bearings_kf_norm, t1 = normalizer_Hartley_non_isotropic(
        x=kwargs["bearings"]["kf"].copy())
    bearings_frm_norm, t2 = normalizer_Hartley_non_isotropic(
        x=kwargs["bearings"]["frm"].copy())

    e_norm, sigma, A = solver.compute_essential_matrix(
        x1=bearings_kf_norm, x2=bearings_frm_norm, return_all=True)

    e_hat = t1.T @ e_norm @ t2
    cam_hat = solver.recover_pose_from_e(
        E=e_hat,
        x1=kwargs["bearings"]['kf'].copy(),
        x2=kwargs["bearings"]['frm'].copy(),
    )
    time_evaluation = time.time() - tic_toc

    paralax_motion = get_angle_between_vectors_arrays(
        bearings_kf_norm, bearings_frm_norm
    )

    return cam_hat, time_evaluation, sigma, np.linalg.norm(A, ord="fro"), paralax_motion


def get_ablation_data_by_Hartley_isotropic(**kwargs):
    tic_toc = time.time()
    bearings_kf_norm, t1 = normalizer_Hartley_isotropic(
        x=kwargs["bearings"]["kf"].copy())
    bearings_frm_norm, t2 = normalizer_Hartley_isotropic(
        x=kwargs["bearings"]["frm"].copy())

    e_norm, sigma, A = solver.compute_essential_matrix(
        x1=bearings_kf_norm, x2=bearings_frm_norm, return_all=True)

    e_hat = t1.T @ e_norm @ t2
    cam_hat = solver.recover_pose_from_e(
        E=e_hat,
        x1=kwargs["bearings"]['kf'].copy(),
        x2=kwargs["bearings"]['frm'].copy(),
    )
    time_evaluation = time.time() - tic_toc

    paralax_motion = get_angle_between_vectors_arrays(
        bearings_kf_norm, bearings_frm_norm
    )

    return cam_hat, time_evaluation, sigma, np.linalg.norm(A, ord="fro"), paralax_motion
