from solvers.epipolar_constraint import EightPointAlgorithmGeneralGeometry as g8p
import levmar
from analysis.utilities.optmization_utilities import *
from solvers.epipolar_constraint import *

solver = g8p()


def get_cam_pose_by_8pa(**kwargs):
    """
    Returns a camera pose using bearing vectors and the 8PA
    """
    cam_pose = solver.recover_pose_from_matches(
        x1=kwargs["bearings"]['kf'].copy(),
        x2=kwargs["bearings"]['frm'].copy(),
        eval_current_solution=True
    )
    return cam_pose, solver.current_residual


def residuals_error_R_T(parameters, bearings_kf, bearings_frm):
    r0 = parameters[0]
    r1 = parameters[1]
    r2 = parameters[2]
    t0 = parameters[3]
    t1 = parameters[4]
    t2 = parameters[5]

    cam_pose = eulerAnglesToRotationMatrix((r0, r1, r2))
    cam_pose[0:3, 3] = np.array((t0, t1, t2)).copy()
    residual = projected_distance(
        e=solver.get_e_from_cam_pose(cam_pose),
        x1=bearings_kf,
        x2=bearings_frm
    )
    return residual


def reprojection_error_R_T(parameters, bearings, landmarks):
    r0 = parameters[0]
    r1 = parameters[1]
    r2 = parameters[2]
    t0 = parameters[3]
    t1 = parameters[4]
    t2 = parameters[5]

    cam_pose = eulerAnglesToRotationMatrix((r0, r1, r2))
    cam_pose[0:3, 3] = np.array((t0, t1, t2)).copy()

    points_hat = np.linalg.inv(cam_pose) @ landmarks
    error = get_projection_error_between_vectors_arrays(
        array_ref=bearings,
        array_vector=points_hat[0:3, :]
    )
    return 1 / error


def get_cam_pose_by_opt_rpj_rt_pnp(**kwargs):
    initial_pose, _ = get_cam_pose_by_8pa(**kwargs)

    landmarks_kf = g8p.triangulate_points_from_cam_pose(
        cam_pose=initial_pose,
        x1=kwargs["bearings"]['kf'].copy(),
        x2=kwargs["bearings"]['frm'].copy(),
    )
    eu = rotationMatrixToEulerAngles(initial_pose[0:3, 0:3])
    trn = np.copy(initial_pose[0:3, 3])

    initial_r_t = np.hstack((eu, trn))

    opt_r_t, p_cov, info = levmar.levmar(
        reprojection_error_R_T,
        initial_r_t,
        np.zeros_like(kwargs["bearings"]["frm"][0, :]),
        args=(kwargs["bearings"]["frm"], landmarks_kf))

    cam_final = eulerAnglesToRotationMatrix(opt_r_t[0:3])
    cam_final[0:3, 3] = opt_r_t[3:].copy()

    print("Opt over Rt by reprojection ")
    print("Iterations: {}".format(info[2]))

    landmarks_frm_hat = np.linalg.inv(cam_final) @ landmarks_kf
    reprojection = get_angle_between_vectors_arrays(
        array_ref=kwargs["bearings"]["frm"].copy(),
        array_vector=landmarks_frm_hat[0:3, :]
    )

    return cam_final, reprojection


def get_cam_pose_by_opt_res_rt_8pa(**kwargs):
    initial_pose, _ = get_cam_pose_by_8pa(**kwargs)
    eu = rotationMatrixToEulerAngles(initial_pose[0:3, 0:3])
    trn = np.copy(initial_pose[0:3, 3])
    initial_r_t = np.hstack((eu, trn))

    opt_r_t, p_cov, info = levmar.levmar(
        residuals_error_R_T,
        initial_r_t,
        np.zeros_like(kwargs["bearings"]["frm"][0, :]),
        args=(kwargs["bearings"]["kf"], kwargs["bearings"]["frm"]))

    print("Opt over Rt by residuals ")
    print("Iterations: {}".format(info[2]))
    cam_final = eulerAnglesToRotationMatrix(opt_r_t[0:3])
    cam_final[0:3, 3] = opt_r_t[3:]
    residual = solver.residual_function_evaluation(
        e=solver.get_e_from_cam_pose(cam_final),
        x1=kwargs["bearings"]['kf'].copy(),
        x2=kwargs["bearings"]['frm'].copy()
    )
    return cam_final, np.sum(residual ** 2)


# * ====================================================================
# * ====================================================================
def mask_bearings_by_pm(all=False, **kwargs):
    if not all:
        delta = np.linalg.norm((kwargs["bearings"]["kf"] - kwargs["bearings"]["frm"]), axis=0)
        max_value = np.max(delta)
        mask = delta > 0.8 * max_value
        if np.sum(mask) < 8:
            mask = delta > 0.5 * max_value
        else:
            mask = delta < max_value
    else:
        delta = np.linalg.norm((kwargs["bearings"]["kf"] - kwargs["bearings"]["frm"]), axis=0)
        mask = delta < np.max(delta) + 1
    return mask


# ! OURS - RES
def res_error_S_K(parameters,
                  bearings_kf, bearings_frm):
    s = parameters[0]
    k = parameters[1]
    bearings_kf_norm, t1 = normalizer_s_k(x=bearings_kf, s=s, k=k)

    # s = parameters[2]
    # k = parameters[3]
    bearings_frm_norm, t2 = normalizer_s_k(x=bearings_frm, s=s, k=k)

    e_norm, sigma, A = solver.compute_essential_matrix(
        x1=bearings_kf_norm,
        x2=bearings_frm_norm,
        return_all=True
    )
    # # ! De-normalization
    # e_hat = t1.T @ e_norm @ t2
    # residuals_error = projected_distance(
    #     e=e_hat,
    #     x1=bearings_kf,
    #     x2=bearings_frm
    # )

    norm_residuals_error = projected_distance(
        e=e_norm,
        x1=bearings_kf_norm,
        x2=bearings_frm_norm
    )
    # return np.array((np.sum((residuals_1 + residuals_2)), sigma[-1] / (sigma[0] - sigma[-2])))
    # return np.array((np.sum(error_1 + residuals_1 + residuals_2),))
    # return residuals_1 + residuals_2
    # return np.array((np.sum(residuals_1), np.sum(residuals_2)))
    # return np.array((np.sum((residuals_1 + residuals_2)), 1 / sigma[-2]))

    return norm_residuals_error
    # return 1 / residuals_error


def get_cam_pose_by_opt_res_error_S_K(**kwargs):
    initial_s_k = (0.001, 0.001)  # initial_k_s = (0.001, 0.001, 0.001, 0.001)
    opt_k_s, p_cov, info = levmar.levmar(
        res_error_S_K,
        initial_s_k,
        np.zeros_like(kwargs["bearings"]["frm"][0, :]),
        # np.array((0, 0)),
        args=(kwargs["bearings"]["kf"].copy(),
              kwargs["bearings"]["frm"].copy()
              ))

    s = opt_k_s[0]
    k = opt_k_s[1]
    print("Opt over KS by residuals")
    print("S:{} K:{}".format(s, k))
    print("Iterations: {}".format(info[2]))
    print("termination: {}".format(info[3]))
    bearings_kf_norm, t1 = normalizer_s_k(
        x=kwargs["bearings"]["kf"].copy(),
        s=s, k=k)
    # s = opt_k_s[2]
    # k = opt_k_s[3]
    # print("S:{} K:{}".format(s, k))
    bearings_frm_norm, t2 = normalizer_s_k(
        x=kwargs["bearings"]["frm"].copy(),
        s=s, k=k)

    e_norm = solver.compute_essential_matrix(
        x1=bearings_kf_norm,
        x2=bearings_frm_norm
    )
    e_hat = t1.T @ e_norm @ t2

    cam_hat = solver.recover_pose_from_e(
        E=e_hat,
        x1=kwargs["bearings"]['kf'].copy(),
        x2=kwargs["bearings"]['frm'].copy(),
    )
    residual = projected_distance(
        e=e_hat,
        x1=kwargs["bearings"]['kf'].copy(),
        x2=kwargs["bearings"]['frm'].copy()
    )
    return cam_hat, np.sum(residual ** 2)


# * ********************************************************************
# ! OURS - REPROJECTION
def rpj_S_K_const_lm(parameters,
                     bearings_kf,
                     bearings_frm,
                     landmarks_kf):
    s = parameters[0]
    k = parameters[1]

    bearings_kf_norm, t1 = normalizer_s_k(x=bearings_kf, s=s, k=k)
    bearings_frm_norm, t2 = normalizer_s_k(x=bearings_frm, s=s, k=k)

    e_norm, sigma, A = solver.compute_essential_matrix(
        x1=bearings_kf_norm,
        x2=bearings_frm_norm,
        return_all=True
    )
    # c_fro_norm = np.linalg.norm(A.T.dot(A), ord="fro")
    e_hat = t1.T @ e_norm @ t2
    cam_hat = solver.recover_pose_from_e(
        E=e_hat,
        x1=bearings_kf,
        x2=bearings_frm
    )
    # landmarks_kf_ = solver.triangulate_points_from_cam_pose(
    #     cam_pose=cam_hat,
    #     x1=bearings_kf.copy(),
    #     x2=bearings_frm.copy()
    # )
    landmarks_frm_hat = np.linalg.inv(cam_hat) @ landmarks_kf
    reprojection_error = get_projection_error_between_vectors_arrays(
        array_ref=bearings_frm,
        array_vector=landmarks_frm_hat[0:3, :]
    )
    # residuals_error = projected_distance(
    #     e=e_hat,
    #     x1=bearings_kf,
    #     x2=bearings_frm
    # )
    #
    norm_residuals_error = projected_distance(
        e=e_norm,
        x1=bearings_kf_norm,
        x2=bearings_frm_norm
    )

    # return np.array((np.sum(1 / error_1), residuals_1, residuals_2))
    # return np.array((np.sum(error_1 + residuals_1 + residuals_2),))
    # return np.array((np.sum((1 / error_1)), (residuals_1 + residuals_2), 1 / sigma[-2]))
    # return norm_residuals_error / np.max(norm_residuals_error)
    return 1 / reprojection_error

    # return (norm_residuals_error * np.linalg.norm(reprojection_error)) / (
    #         reprojection_error * np.linalg.norm(norm_residuals_error))


def get_cam_pose_by_opt_rpj_S_K_const_lm(**kwargs):
    initial_pose, _ = get_cam_pose_by_8pa(**kwargs)
    # initial_s_k = (0.1, 0.1)
    # initial_k_s = (0.001, 0.001, 0.001, 0.001)
    initial_s_k = (0.001, 0.001)

    landmarks_kf = solver.triangulate_points_from_cam_pose(
        cam_pose=initial_pose,
        x1=kwargs["bearings"]['kf'].copy(),
        x2=kwargs["bearings"]['frm'].copy(),
    )
    opt_k_s, p_cov, info = levmar.levmar(
        rpj_S_K_const_lm,
        initial_s_k,
        np.zeros_like(kwargs["bearings"]["frm"][0, :]),
        # np.array((0, 0, 0)),
        args=(kwargs["bearings"]["kf"].copy(),
              kwargs["bearings"]["frm"].copy(),
              landmarks_kf
              ))

    s = opt_k_s[0]
    k = opt_k_s[1]
    print("Opt over KS by reprojection")
    print("S:{} K:{}".format(s, k))
    bearings_kf_norm, t1 = normalizer_s_k(
        x=kwargs["bearings"]["kf"].copy(),
        s=s, k=k)
    # s = opt_k_s[2]
    # k = opt_k_s[3]
    # print("S:{} K:{}".format(s, k))
    print("iterations: {}".format(info[2]))
    print("termination: {}".format(info[3]))

    bearings_frm_norm, t2 = normalizer_s_k(
        x=kwargs["bearings"]["frm"].copy(),
        s=s, k=k)

    e_norm = solver.compute_essential_matrix(
        x1=bearings_kf_norm,
        x2=bearings_frm_norm
    )
    e_hat = t2.T @ e_norm @ t1

    cam_hat = solver.recover_pose_from_e(
        E=e_hat,
        x1=kwargs["bearings"]['kf'].copy(),
        x2=kwargs["bearings"]['frm'].copy(),
    )
    landmarks_kf = solver.triangulate_points_from_cam_pose(
        cam_pose=cam_hat,
        x1=kwargs["bearings"]['kf'].copy(),
        x2=kwargs["bearings"]['frm'].copy(),
    )

    landmarks_frm_hat = np.linalg.inv(cam_hat) @ landmarks_kf
    reprojection = get_angle_between_vectors_arrays(
        array_ref=kwargs["bearings"]["frm"].copy(),
        array_vector=landmarks_frm_hat[0:3, :]
    )

    return cam_hat, reprojection
