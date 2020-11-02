from solvers.epipolar_constraint import EightPointAlgorithmGeneralGeometry as g8p
import levmar
from analysis.utilities.optmization_utilities import *
from solvers.epipolar_constraint import *
import time
from liegroups.numpy import SE3
from scipy.stats import norm
from scipy.stats import t

solver = g8p()


def get_cam_pose_by_8pa(**kwargs):
    """
    Returns a camera pose using bearing vectors and the 8PA
    """
    time_evaluation = time.time()
    cam_pose = solver.recover_pose_from_matches(
        x1=kwargs["bearings"]['kf'].copy(),
        x2=kwargs["bearings"]['frm'].copy(),
    )
    time_evaluation = time.time() - time_evaluation
    residual = projected_error(
        e=solver.get_e_from_cam_pose(cam_pose),
        x1=kwargs["bearings"]['kf'].copy(),
        x2=kwargs["bearings"]['frm'].copy())
    return cam_pose, residual, time_evaluation


# * ****************************************************************
# ! Optimization by the Gold Standard Method (GSM)
def residuals_error_Rt_L2(parameters, bearings_kf, bearings_frm):
    """
    Loss function for opt(R,t)
    """
    cam_pose = SE3.exp(parameters).as_matrix()
    residual = projected_error(
        e=solver.get_e_from_cam_pose(cam_pose),
        x1=bearings_kf,
        x2=bearings_frm)
    # * RT: L2
    loss = residual
    return loss


def get_cam_pose_by_opt_Rt_L2(**kwargs):
    initial_pose = get_cam_pose_by_8pa(**kwargs)[0]
    tic_toc = time.time()
    xi = SE3.log(SE3.from_matrix(initial_pose))
    opt_r_t, p_cov, info = levmar.levmar(
        func=residuals_error_Rt_L2,
        p0=xi,
        y=np.zeros_like(kwargs["bearings"]["frm"][0, :]),
        args=(kwargs["bearings"]["kf"], kwargs["bearings"]["frm"]))

    cam_final = SE3.exp(opt_r_t).as_matrix()
    final_time = solver.timing + time.time() - tic_toc
    print("***********************************")
    print("Opt RESIDUALS over Rt ")
    print("Iterations: {}".format(info[2]))
    print("termination: {}".format(info[3]))
    residual = projected_error(
        e=solver.get_e_from_cam_pose(cam_final),
        x1=kwargs["bearings"]['kf'].copy(),
        x2=kwargs["bearings"]['frm'].copy())
    if kwargs.get("timing_evaluation", False):
        return cam_final, np.sum(residual ** 2), final_time
    return cam_final, np.sum(residual ** 2)


# * ====================================================================
# * ====================================================================
# * ====================================================================
# * ====================================================================


def residuals_error_Rt_L1(parameters, bearings_kf, bearings_frm):
    cam_pose = SE3.exp(parameters).as_matrix()

    residual = projected_error(
        e=solver.get_e_from_cam_pose(cam_pose),
        x1=bearings_kf,
        x2=bearings_frm)

    # # * RT: *L1
    loss = np.zeros_like(bearings_kf[0, :])
    # ! It is better without sqrt
    loss[0] = np.sum(np.abs(residual))
    return loss


def get_cam_pose_by_opt_Rt_L1(**kwargs):
    initial_pose = get_cam_pose_by_8pa(**kwargs)[0]
    tic_toc = time.time()
    xi = SE3.log(SE3.from_matrix(initial_pose))
    opt_r_t, p_cov, info = levmar.levmar(
        func=residuals_error_Rt_L1,
        p0=xi,
        y=np.zeros_like(kwargs["bearings"]["frm"][0, :]),
        args=(kwargs["bearings"]["kf"], kwargs["bearings"]["frm"]))

    final_time = solver.timing + time.time() - tic_toc
    print("***********************************")
    print("Opt RESIDUALS over Rt ")
    print("Iterations: {}".format(info[2]))
    print("termination: {}".format(info[3]))

    cam_final = SE3.exp(opt_r_t).as_matrix()

    residual = projected_error(
        e=solver.get_e_from_cam_pose(cam_final),
        x1=kwargs["bearings"]['kf'].copy(),
        x2=kwargs["bearings"]['frm'].copy())
    if kwargs.get("timing_evaluation", False):
        return cam_final, np.sum(residual ** 2), final_time
    return cam_final, np.sum(residual ** 2)


# * ====================================================================
# * ====================================================================
# * ====================================================================
# * ====================================================================
# ! OURS - OPT SK (residuals)
def residuals_error_SK(parameters, bearings_kf, bearings_frm):
    s = parameters[0]
    k = parameters[1]
    bearings_kf_norm, t1 = normalizer_s_k(x=bearings_kf, s=s, k=k)
    bearings_frm_norm, t2 = normalizer_s_k(x=bearings_frm, s=s, k=k)

    e_norm, sigma, A = solver.compute_essential_matrix(
        x1=bearings_kf_norm, x2=bearings_frm_norm, return_all=True)
    # # # ! De-normalization
    e_hat = t1.T @ e_norm @ t2
    residuals_error = projected_error(
        e=e_hat, x1=bearings_kf, x2=bearings_frm)

    # * KS: L1
    loss = np.zeros_like(parameters)
    loss[0] = np.sum(np.abs(residuals_error))

    return loss


def get_cam_pose_by_opt_SK(**kwargs):
    initial_s_k = kwargs[
        "iVal_Res_SK"]

    tic_toc = time.time()
    opt_k_s, p_cov, info = levmar.levmar(
        func=residuals_error_SK,
        p0=initial_s_k,
        y=np.zeros_like(initial_s_k),
        args=(kwargs["bearings"]["kf"].copy(),
              kwargs["bearings"]["frm"].copy()))

    delta_time = time.time() - tic_toc
    s = opt_k_s[0]
    k = opt_k_s[1]
    print("***********************************")
    print("** OURS RESIDUALS ** Opt over KS ")
    print("Initials S:{} K:{}".format(initial_s_k[0], initial_s_k[1]))
    print("S:{} K:{}".format(s, k))
    print("Iterations: {}".format(info[2]))
    print("termination: {}".format(info[3]))
    tic_toc = time.time()
    bearings_kf_norm, t1 = normalizer_s_k(
        x=kwargs["bearings"]["kf"].copy(), s=s, k=k)
    bearings_frm_norm, t2 = normalizer_s_k(
        x=kwargs["bearings"]["frm"].copy(), s=s, k=k)

    e_norm = solver.compute_essential_matrix(
        x1=bearings_kf_norm, x2=bearings_frm_norm)
    e_hat = t1.T @ e_norm @ t2
    cam_hat = solver.recover_pose_from_e(
        E=e_hat,
        x1=kwargs["bearings"]['kf'].copy(),
        x2=kwargs["bearings"]['frm'].copy(),
    )
    delta_time += time.time() - tic_toc

    residual = projected_error(
        e=e_hat,
        x1=kwargs["bearings"]['kf'].copy(),
        x2=kwargs["bearings"]['frm'].copy())
    if kwargs.get("timing_evaluation", False):
        return cam_hat, np.sum(residual ** 2), delta_time
    return cam_hat, np.sum(residual ** 2)


#
# # * ********************************************************************
# # * ********************************************************************
#

# # * ********************************************************************
# # * ********************************************************************
# # ! ********************************************************************
# # ! ********************************************************************
# # ! ********************************************************************
# # * ********************************************************************
# # * ********************************************************************

def residuals_error_KS_wRt_L2(parameters, bearings_kf, bearings_frm, s, k):
    # ! OURS
    bearings_kf_norm, n1 = normalizer_s_k(x=bearings_kf, s=s, k=k)
    bearings_frm_norm, n2 = normalizer_s_k(x=bearings_frm, s=s, k=k)

    cam_pose = SE3.exp(parameters).as_matrix()

    e_rt = solver.get_e_from_cam_pose(cam_pose)

    e_norm = np.linalg.inv(n1).T @ e_rt @ np.linalg.inv(n2)

    residuals_error = projected_error(
        e=e_rt, x1=bearings_kf, x2=bearings_frm)

    norm_residuals_error = algebraic_error(
        e=e_norm, x1=bearings_kf_norm, x2=bearings_frm_norm)

    weights = norm(np.mean(norm_residuals_error), np.std(norm_residuals_error))
    # weights = t.pdf(norm_residuals_error, df=5)
    # * KS-wRT: L2
    loss = np.sqrt(weights.pdf(norm_residuals_error)) * residuals_error
    # loss = np.sqrt(weights) * residuals_error

    return loss


# ! OURS
def get_cam_pose_by_opt_SK_wRt_L2(**kwargs):
    initial_s_k = kwargs[
        "iVal_Res_SK"]

    initial_time = time.time()
    opt_k_s, p_cov, info = levmar.levmar(
        func=residuals_error_SK,
        p0=initial_s_k,
        y=np.zeros_like(initial_s_k),
        args=(kwargs["bearings"]["kf"].copy(),
              kwargs["bearings"]["frm"].copy(),))

    s = opt_k_s[0]
    k = opt_k_s[1]
    bearings_kf_norm, t1 = normalizer_s_k(
        x=kwargs["bearings"]["kf"].copy(), s=s, k=k)
    bearings_frm_norm, t2 = normalizer_s_k(
        x=kwargs["bearings"]["frm"].copy(), s=s, k=k)

    e_norm = solver.compute_essential_matrix(
        x1=bearings_kf_norm, x2=bearings_frm_norm)
    e_hat = t1.T @ e_norm @ t2

    cam_hat = solver.recover_pose_from_e(
        E=e_hat,
        x1=kwargs["bearings"]['kf'].copy(),
        x2=kwargs["bearings"]['frm'].copy(),
    )

    delta_time = time.time() - initial_time
    print("***********************************")
    print("** OURS RESIDUALS ** Opt over KS-Rt ")
    print("Initials S:{} K:{}".format(initial_s_k[0], initial_s_k[1]))
    print("S:{} K:{}".format(s, k))
    print("Iterations: {}".format(info[2]))
    print("termination: {}".format(info[3]))

    initial_time = time.time()
    xi = SE3.log(SE3.from_matrix(cam_hat))
    opt_rt, p_cov, info = levmar.levmar(
        func=residuals_error_KS_wRt_L2,
        y=np.zeros_like(kwargs["bearings"]["kf"][0, :]),
        p0=xi,
        args=(kwargs["bearings"]["kf"].copy(),
              kwargs["bearings"]["frm"].copy(),
              s, k
              ))
    cam_final = SE3.exp(opt_rt).as_matrix()
    delta_time = time.time() - initial_time + delta_time

    print("Iterations: {}".format(info[2]))
    print("termination: {}".format(info[3]))

    residual = projected_error(
        e=solver.get_e_from_cam_pose(cam_final),
        x1=kwargs["bearings"]['kf'].copy(),
        x2=kwargs["bearings"]['frm'].copy())

    if kwargs.get("timing_evaluation", False):
        return cam_final, np.sum(residual ** 2), delta_time
    return cam_final, np.sum(residual ** 2)


# # ! ********************************************************************
# # * ********************************************************************
# ! OURS
def residuals_error_const_wRT_L2(parameters, bearings_kf, bearings_frm, weights):
    cam_pose = SE3.exp(parameters).as_matrix()

    e_rt = solver.get_e_from_cam_pose(cam_pose)

    residuals_error = projected_error(
        e=e_rt, x1=bearings_kf, x2=bearings_frm)

    # # * KS-wRT: L2
    loss = np.sqrt(weights) * residuals_error
    return loss


# ! OURS
def get_cam_pose_by_opt_SK_const_wRt_L2(**kwargs):
    initial_s_k = kwargs[
        "iVal_Res_SK"]
    initial_time = time.time()
    opt_k_s, p_cov, info = levmar.levmar(
        func=residuals_error_SK,
        p0=initial_s_k,
        y=np.zeros_like(initial_s_k),
        args=(kwargs["bearings"]["kf"].copy(),
              kwargs["bearings"]["frm"].copy(),))

    s = opt_k_s[0]
    k = opt_k_s[1]
    bearings_kf_norm, t1 = normalizer_s_k(
        x=kwargs["bearings"]["kf"].copy(), s=s, k=k)
    bearings_frm_norm, t2 = normalizer_s_k(
        x=kwargs["bearings"]["frm"].copy(), s=s, k=k)

    e_norm = solver.compute_essential_matrix(
        x1=bearings_kf_norm, x2=bearings_frm_norm)
    e_hat = t1.T @ e_norm @ t2

    cam_hat = solver.recover_pose_from_e(
        E=e_hat,
        x1=kwargs["bearings"]['kf'].copy(),
        x2=kwargs["bearings"]['frm'].copy(),
    )

    delta_time = time.time() - initial_time
    print("***********************************")
    print("** OURS RESIDUALS ** Opt over KS-Rt ")
    print("Initials S:{} K:{}".format(initial_s_k[0], initial_s_k[1]))
    print("S:{} K:{}".format(s, k))
    print("Iterations: {}".format(info[2]))
    print("termination: {}".format(info[3]))

    initial_time = time.time()
    norm_residuals_error = algebraic_error(e=e_norm, x1=bearings_kf_norm, x2=bearings_frm_norm)
    residual = projected_error(
        e=e_hat,
        x1=kwargs["bearings"]['kf'].copy(),
        x2=kwargs["bearings"]['frm'].copy())
    weights = norm(np.mean(residual), np.std(residual))
    #
    # weights = t.pdf(residual, df=5)

    xi = SE3.log(SE3.from_matrix(cam_hat))
    opt_rt, p_cov, info = levmar.levmar(
        func=residuals_error_const_wRT_L2,
        y=np.zeros_like(kwargs["bearings"]["kf"][0, :]),
        p0=xi,
        args=(kwargs["bearings"]["kf"].copy(),
              kwargs["bearings"]["frm"].copy(),
              # s, k
              weights.pdf(residual)
              # weights
              ))
    cam_final = SE3.exp(opt_rt).as_matrix()
    delta_time = time.time() - initial_time + delta_time

    print("Iterations: {}".format(info[2]))
    print("termination: {}".format(info[3]))

    residual = projected_error(
        e=solver.get_e_from_cam_pose(cam_final),
        x1=kwargs["bearings"]['kf'].copy(),
        x2=kwargs["bearings"]['frm'].copy())

    if kwargs.get("timing_evaluation", False):
        return cam_final, np.sum(residual ** 2), delta_time
    return cam_final, np.sum(residual ** 2)


# # * ********************************************************************
# # * ********************************************************************
# # ? ********************************************************************
# # ? ********************************************************************
# # ? ********************************************************************
# # * ********************************************************************
# # * ********************************************************************

def residuals_error_wRt_L2(parameters, bearings_kf, bearings_frm):
    cam_pose = SE3.exp(parameters).as_matrix()

    e_rt = solver.get_e_from_cam_pose(cam_pose)

    residuals_error = projected_error(
        e=e_rt, x1=bearings_kf, x2=bearings_frm)

    # ! residuals
    # weights = t.pdf(residuals_error, df=5)
    weights = norm(np.mean(residuals_error), np.std(residuals_error))

    # * KS-wRT: L2
    loss = np.sqrt(weights.pdf(residuals_error)) * residuals_error
    # loss = np.sqrt(weights) * residuals_error

    return loss


def get_cam_pose_by_opt_wRt_L2(**kwargs):
    initial_time = time.time()
    initial_pose, _, _ = get_cam_pose_by_8pa(**kwargs)
    xi = SE3.log(SE3.from_matrix(initial_pose))
    opt_rt, p_cov, info = levmar.levmar(
        func=residuals_error_wRt_L2,
        y=np.zeros_like(kwargs["bearings"]["kf"][0, :]),
        p0=xi,
        args=(kwargs["bearings"]["kf"].copy(),
              kwargs["bearings"]["frm"].copy(),
              ))
    cam_final = SE3.exp(opt_rt).as_matrix()
    delta_time = time.time() - initial_time

    print("Iterations: {}".format(info[2]))
    print("termination: {}".format(info[3]))

    residual = projected_error(
        e=solver.get_e_from_cam_pose(cam_final),
        x1=kwargs["bearings"]['kf'].copy(),
        x2=kwargs["bearings"]['frm'].copy())

    if kwargs.get("timing_evaluation", False):
        return cam_final, np.sum(residual ** 2), delta_time
    return cam_final, np.sum(residual ** 2)


# # ! ********************************************************************
# # * ********************************************************************

def get_cam_pose_by_opt_const_wRt_L2(**kwargs):
    # ! residuals
    initial_time = time.time()
    cam_hat, _, _ = get_cam_pose_by_8pa(**kwargs)
    residual = projected_error(
        e=solver.get_e_from_cam_pose(cam_hat),
        x1=kwargs["bearings"]['kf'].copy(),
        x2=kwargs["bearings"]['frm'].copy())

    weights = norm(np.mean(residual), np.std(residual))
    # weights = t.pdf(residual, df=5)

    xi = SE3.log(SE3.from_matrix(cam_hat))
    opt_rt, p_cov, info = levmar.levmar(
        func=residuals_error_const_wRT_L2,
        y=np.zeros_like(kwargs["bearings"]["kf"][0, :]),
        p0=xi,
        args=(kwargs["bearings"]["kf"].copy(),
              kwargs["bearings"]["frm"].copy(),
              weights.pdf(residual)
              # weights
              ))
    cam_final = SE3.exp(opt_rt).as_matrix()

    delta_time = time.time() - initial_time

    print("Iterations: {}".format(info[2]))
    print("termination: {}".format(info[3]))

    residual = projected_error(
        e=solver.get_e_from_cam_pose(cam_final),
        x1=kwargs["bearings"]['kf'].copy(),
        x2=kwargs["bearings"]['frm'].copy())

    if kwargs.get("timing_evaluation", False):
        return cam_final, np.sum(residual ** 2), delta_time
    return cam_final, np.sum(residual ** 2)
