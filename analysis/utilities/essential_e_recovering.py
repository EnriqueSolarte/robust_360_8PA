from analysis.utilities.camera_recovering import *

solver = g8p()


def get_e_by_8pa(**kwargs):
    """
    Returns the essential matrix by using the classic 8PA [Higgins'1981]
    """
    e_hat = solver.compute_essential_matrix(
        x1=kwargs["bearings"]['kf'].copy(),
        x2=kwargs["bearings"]['frm'].copy(),
    )
    return e_hat


def get_e_by_opt_Rt_L2(**kwargs):
    # ! Initial R|t is needed in advance
    initial_pose = get_cam_pose_by_8pa(**kwargs)[0]
    xi = SE3.log(SE3.from_matrix(initial_pose))
    # ! LM-optimization
    opt_r_t, p_cov, info = levmar.levmar(
        residuals_error_Rt_L2,
        xi,
        np.zeros_like(kwargs["bearings"]["frm"][0, :]),
        args=(kwargs["bearings"]["kf"], kwargs["bearings"]["frm"]))

    cam_final = SE3.exp(opt_r_t).as_matrix()
    return solver.get_e_from_cam_pose(cam_pose=cam_final)


# * ====================================================================
# * ====================================================================
# * ====================================================================
# * ====================================================================
# * ====================================================================
# * ====================================================================
# * ====================================================================
# * ====================================================================
def get_e_by_opt_SK(**kwargs):
    # ! Initial K,S parameters
    initial_s_k = kwargs[
        "iVal_Res_SK"]

    opt_k_s, p_cov, info = levmar.levmar(
        func=residuals_error_SK,
        p0=initial_s_k,
        y=np.zeros_like(initial_s_k),
        args=(kwargs["bearings"]["kf"].copy(),
              kwargs["bearings"]["frm"].copy()))

    s = opt_k_s[0]
    k = opt_k_s[1]
    bearings_kf_norm, t1 = normalizer_s_k(
        x=kwargs["bearings"]["kf"].copy(), s=s, k=k)
    bearings_frm_norm, t2 = normalizer_s_k(
        x=kwargs["bearings"]["frm"].copy(), s=s, k=k)

    e_norm = solver.compute_essential_matrix(
        x1=bearings_kf_norm, x2=bearings_frm_norm)
    e_hat = t1.T @ e_norm @ t2
    return e_hat


# # * ********************************************************************
# ? * ********************************************************************
# # * ********************************************************************

def get_e_by_opt_SK_wRt_L2(**kwargs):
    initial_s_k = kwargs[
        "iVal_Res_SK"]

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
    return solver.get_e_from_cam_pose(cam_final)


def get_e_by_opt_SK_const_wRt_L2(**kwargs):
    initial_s_k = kwargs[
        "iVal_Res_SK"]
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
    residual = projected_error(
        e=e_hat,
        x1=kwargs["bearings"]['kf'].copy(),
        x2=kwargs["bearings"]['frm'].copy())
    weights = norm(np.mean(residual), np.std(residual))

    xi = SE3.log(SE3.from_matrix(cam_hat))
    opt_rt, p_cov, info = levmar.levmar(
        func=residuals_error_const_wRT_L2,
        y=np.zeros_like(kwargs["bearings"]["kf"][0, :]),
        p0=xi,
        args=(kwargs["bearings"]["kf"].copy(),
              kwargs["bearings"]["frm"].copy(),
              weights.pdf(residual)
              ))
    cam_final = SE3.exp(opt_rt).as_matrix()
    return solver.get_e_from_cam_pose(cam_final)


def get_e_by_opt_wRt_L2(**kwargs):
    initial_pose = get_cam_pose_by_8pa(**kwargs)[0]
    xi = SE3.log(SE3.from_matrix(initial_pose))
    opt_rt, p_cov, info = levmar.levmar(
        func=residuals_error_wRt_L2,
        y=np.zeros_like(kwargs["bearings"]["kf"][0, :]),
        p0=xi,
        args=(kwargs["bearings"]["kf"].copy(),
              kwargs["bearings"]["frm"].copy(),
              ))

    cam_final = SE3.exp(opt_rt).as_matrix()
    return solver.get_e_from_cam_pose(cam_final)


def get_e_by_opt_const_wRt_L2(**kwargs):
    cam_hat = get_cam_pose_by_8pa(**kwargs)[0]
    residual = projected_error(
        e=solver.get_e_from_cam_pose(cam_hat),
        x1=kwargs["bearings"]['kf'].copy(),
        x2=kwargs["bearings"]['frm'].copy())

    weights = norm(np.mean(residual), np.std(residual))

    xi = SE3.log(SE3.from_matrix(cam_hat))
    opt_rt, p_cov, info = levmar.levmar(
        func=residuals_error_const_wRT_L2,
        y=np.zeros_like(kwargs["bearings"]["kf"][0, :]),
        p0=xi,
        args=(kwargs["bearings"]["kf"].copy(),
              kwargs["bearings"]["frm"].copy(),
              weights.pdf(residual)
              ))
    cam_final = SE3.exp(opt_rt).as_matrix()
    return solver.get_e_from_cam_pose(cam_final)
