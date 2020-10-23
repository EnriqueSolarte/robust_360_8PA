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


def get_e_by_opt_res_error_Rt_L2(**kwargs):
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
def get_e_by_opt_res_error_SK(**kwargs):
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

def get_e_by_opt_res_error_SKRt(**kwargs):
    initial_pose = get_cam_pose_by_8pa(**kwargs)[0]
    initial_s_k = kwargs["iVal_Res_SK"]

    xi = SE3.log(SE3.from_matrix(initial_pose))
    initial_rtsk = np.hstack((xi, initial_s_k))

    opt_rt, p_cov, info = levmar.levmar(
        func=residuals_error_RtKS,
        p0=initial_rtsk,
        y=np.zeros_like(kwargs["bearings"]["kf"][0, :]),
        args=(kwargs["bearings"]["kf"].copy(),
              kwargs["bearings"]["frm"].copy()
              ))
    cam_final = SE3.exp(opt_rt[0:6]).as_matrix()
    return solver.get_e_from_cam_pose(cam_pose=cam_final)


# * ********************************************************************


def get_e_by_opt_res_error_RtKS(**kwargs):
    initial_s_k = kwargs[
        "iVal_Res_SK"]

    opt_k_s, p_cov, info = levmar.levmar(
        func=residuals_error_SK,
        p0=initial_s_k,
        y=np.array((0, 0, 0, 0)),
        maxit=500,
        args=(kwargs["bearings"]["kf"].copy(),
              kwargs["bearings"]["frm"].copy()))

    s = opt_k_s[0]
    k = opt_k_s[1]

    # ! To speed up, we are using a previous e matrix saved at the
    # ! LM loop optimization
    # initial_pose = get_cam_pose_by_8pa(**kwargs)[0]
    initial_pose = solver.recover_pose_from_e(
        E=solver.current_e_matrix,
        x1=kwargs["bearings"]["kf"].copy(),
        x2=kwargs["bearings"]["frm"].copy()
    )
    eu = rotationMatrixToEulerAngles(initial_pose[0:3, 0:3])
    trn = np.copy(initial_pose[0:3, 3])

    initial_rt = np.hstack((eu, trn))
    opt_rt, p_cov, info = levmar.levmar(
        func=residuals_error_KSRT,
        p0=initial_rt,
        y=np.zeros_like(initial_rt),
        maxit=500,
        args=(kwargs["bearings"]["kf"].copy(),
              kwargs["bearings"]["frm"].copy(), s, k))

    cam_final = eulerAnglesToRotationMatrix(opt_rt[0:3])
    cam_final[0:3, 3] = opt_rt[3:6].copy()
    return solver.get_e_from_cam_pose(cam_pose=cam_final)
# * ********************************************************************
