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


def residuals_error_R_T(parameters, bearings_kf, bearings_frm):
    r0 = parameters[0]
    r1 = parameters[1]
    r2 = parameters[2]
    t0 = parameters[3]
    t1 = parameters[4]
    t2 = parameters[5]

    cam_pose = eulerAnglesToRotationMatrix((r0, r1, r2))
    cam_pose[0:3, 3] = np.array((t0, t1, t2)).copy()
    residual = projected_error(
        e=solver.get_e_from_cam_pose(cam_pose),
        x1=bearings_kf,
        x2=bearings_frm)
    return residual


def get_e_by_opt_res_error_Rt(**kwargs):
    # ! Initial R|t is needed in advance
    initial_pose = get_cam_pose_by_8pa(**kwargs)[0]

    eu = rotationMatrixToEulerAngles(initial_pose[0:3, 0:3])
    trn = np.copy(initial_pose[0:3, 3])
    initial_r_t = np.hstack((eu, trn))

    # ! LM-optimization
    opt_r_t, p_cov, info = levmar.levmar(
        residuals_error_R_T,
        initial_r_t,
        np.zeros_like(kwargs["bearings"]["frm"][0, :]),
        args=(kwargs["bearings"]["kf"], kwargs["bearings"]["frm"]))

    cam_final = eulerAnglesToRotationMatrix(opt_r_t[0:3])
    cam_final[0:3, 3] = opt_r_t[3:]
    return solver.get_e_from_cam_pose(cam_pose=cam_final)


# * ====================================================================
# * ====================================================================
# * ====================================================================
# * ====================================================================
# * ====================================================================
# * ====================================================================
# * ====================================================================
# * ====================================================================
# ! OURS - OPT SK (residuals)
def res_error_S_K(parameters, bearings_kf, bearings_frm):
    s = parameters[0]
    k = parameters[1]
    bearings_kf_norm, t1 = normalizer_s_k(x=bearings_kf, s=s, k=k)
    bearings_frm_norm, t2 = normalizer_s_k(x=bearings_frm, s=s, k=k)

    e_norm, sigma, A = solver.compute_essential_matrix(
        x1=bearings_kf_norm, x2=bearings_frm_norm, return_all=True)
    C = np.linalg.norm(A.T.dot(A), ord="fro")

    #  ! De-normalization
    # TODO We can save this in solver
    e_hat = t1.T @ e_norm @ t2
    solver.current_e_matrix = e_hat
    residuals_error = projected_error(
        e=e_hat, x1=bearings_kf, x2=bearings_frm)

    norm_residuals_error = algebraic_error(
        e=e_norm, x1=bearings_kf_norm, x2=bearings_frm_norm)

    loss = np.zeros((4,))
    loss[0] = np.sum(residuals_error)
    loss[1] = np.sum(norm_residuals_error)
    loss[2] = sigma[-1]
    loss[3] = C ** 2
    return loss


def get_cam_pose_by_opt_res_error_SK(**kwargs):
    # ! Initial K,S parameters
    initial_s_k = kwargs[
        "iVal_Res_SK"]

    opt_k_s, p_cov, info = levmar.levmar(
        func=res_error_S_K,
        p0=initial_s_k,
        # maxit=100,
        y=np.array((0, 0, 0, 0)),
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


# ! OPTION 2 - OPt KS_RT
def residuals_error_KS_RT(parameters, bearings_kf, bearings_frm, s, k):
    r0 = parameters[0]
    r1 = parameters[1]
    r2 = parameters[2]
    t0 = parameters[3]
    t1 = parameters[4]
    t2 = parameters[5]

    bearings_kf_norm, n1 = normalizer_s_k(x=bearings_kf, s=s, k=k)
    bearings_frm_norm, n2 = normalizer_s_k(x=bearings_frm, s=s, k=k)

    cam_pose = eulerAnglesToRotationMatrix((r0, r1, r2))
    cam_pose[0:3, 3] = np.array((t0, t1, t2)).copy()
    e_rt = solver.get_e_from_cam_pose(cam_pose)
    e_norm = np.linalg.inv(n1).T @ e_rt @ np.linalg.inv(n2)

    residuals_error = projected_error(
        e=e_rt, x1=bearings_kf, x2=bearings_frm)

    norm_residuals_error = algebraic_error(
        e=e_norm, x1=bearings_kf_norm, x2=bearings_frm_norm)

    loss = np.zeros_like(parameters)
    loss[1] = np.sum(residuals_error)
    loss[0] = np.sum(norm_residuals_error)
    return loss


def get_e_by_opt_res_error_SK_Rt(**kwargs):
    initial_s_k = kwargs[
        "iVal_Res_SK"]  # initial_k_s = (0.001, 0.001, 0.001, 0.001)

    opt_k_s, p_cov, info = levmar.levmar(
        func=res_error_S_K,
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
        func=residuals_error_KS_RT,
        p0=initial_rt,
        y=np.zeros_like(initial_rt),
        maxit=500,
        args=(kwargs["bearings"]["kf"].copy(),
              kwargs["bearings"]["frm"].copy(), s, k))

    cam_final = eulerAnglesToRotationMatrix(opt_rt[0:3])
    cam_final[0:3, 3] = opt_rt[3:6].copy()
    return solver.get_e_from_cam_pose(cam_pose=cam_final)
# * ********************************************************************
