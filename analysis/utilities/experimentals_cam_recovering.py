from solvers.epipolar_constraint import EightPointAlgorithmGeneralGeometry as g8p
import levmar
from solvers.epipolar_constraint import *
from analysis.utilities.optmization_utilities import normalizer_rsk, normalizer_Cxyz

solver = g8p()


# * ********************************************************************

# ! OPt RKS
def residuals_error_RKS(parameters, bearings_kf, bearings_frm):
    r = parameters[0:3]

    k = parameters[3]
    s = parameters[4]

    bearings_kf_norm, n1 = normalizer_rsk(
        x=bearings_kf, s=s, k=k, r=r)
    bearings_frm_norm, n2 = normalizer_rsk(
        x=bearings_frm, s=s, k=k, r=r)

    e_norm, sigma, A = solver.compute_essential_matrix(
        x1=bearings_kf_norm,
        x2=bearings_frm_norm,
        return_all=True
    )

    norm_residuals_error = projected_distance(
        e=e_norm,
        x1=bearings_kf_norm,
        x2=bearings_frm_norm
    )
    # e_hat = n1.T @ e_norm @ n1
    # residuals_error = projected_distance(
    #     e=e_hat,
    #     x1=bearings_kf,
    #     x2=bearings_frm
    # )
    return norm_residuals_error


def get_cam_pose_by_opt_res_error_RSK(**kwargs):
    initial_s_k = kwargs["iVal_Res_RSK"]
    eu = np.zeros((3,))
    initial_rsk = np.hstack((eu, initial_s_k))
    opt_rt_sk, p_cov, info = levmar.levmar(
        residuals_error_RKS,
        initial_rsk,
        np.zeros_like(kwargs["bearings"]["frm"][0, :]),
        # np.array((0, 0, 0)),
        args=(kwargs["bearings"]["kf"].copy(),
              kwargs["bearings"]["frm"].copy(),
              ))

    s = opt_rt_sk[3]
    k = opt_rt_sk[4]
    r = opt_rt_sk[0:3]

    print("***********************************")
    print("** OURS RESIDUALS ** Opt over RKS <<<<<<< ")
    print("Initials S:{} K:{} R:{}".format(s, k, r))
    print("S:{} K:{}".format(s, k))
    print("Iterations: {}".format(info[2]))
    print("termination: {}".format(info[3]))
    residuals = residuals_error_RKS(
        parameters=opt_rt_sk,
        bearings_kf=kwargs["bearings"]["kf"].copy(),
        bearings_frm=kwargs["bearings"]["frm"].copy(),
    )

    bearings_kf_norm, t1 = normalizer_rsk(
        x=kwargs["bearings"]["kf"].copy(),
        s=s, k=k, r=r)

    bearings_frm_norm, t2 = normalizer_rsk(
        x=kwargs["bearings"]["frm"].copy(),
        s=s, k=k, r=r)

    e_norm = solver.compute_essential_matrix(
        x1=bearings_kf_norm,
        x2=bearings_frm_norm
    )
    e_hat = t2.T @ e_norm @ t1

    cam_final = solver.recover_pose_from_e(
        E=e_hat,
        x1=kwargs["bearings"]['kf'].copy(),
        x2=kwargs["bearings"]['frm'].copy(),
    )

    return cam_final, np.sum(residuals ** 2)



# ! OPt Cxyz
def residuals_error_Cxyz(parameters, bearings_kf, bearings_frm):
    bearings_kf_norm, n1 = normalizer_Cxyz(x=bearings_kf, Cxyz=parameters)
    bearings_frm_norm, n2 = normalizer_Cxyz(x=bearings_frm, Cxyz=parameters)

    e_norm, sigma, A = solver.compute_essential_matrix(
        x1=bearings_kf_norm,
        x2=bearings_frm_norm,
        return_all=True
    )
    c_fro_norm = np.linalg.norm(A.T.dot(A), ord="fro")

    e_hat = n1.T @ e_norm @ n2
    residuals_error = projected_distance(
        e=e_hat,
        x1=bearings_kf,
        x2=bearings_frm
    )

    norm_residuals_error = sampson_distance(
        e=e_norm,
        x1=bearings_kf_norm,
        x2=bearings_frm_norm
    )

    return c_fro_norm * residuals_error


def get_cam_pose_by_opt_res_error_Cxyz(**kwargs):
    initial_Cxyz = kwargs["iVal_Res_Cxyz"]

    opt_Cxyz, p_cov, info = levmar.levmar(
        residuals_error_Cxyz,
        initial_Cxyz,
        np.zeros_like(kwargs["bearings"]["frm"][0, :]),
        # np.array((0, 0, 0)),
        args=(kwargs["bearings"]["kf"].copy(),
              kwargs["bearings"]["frm"].copy(),
              ),
        cdiff=True)

    print("***********************************")
    print("** OURS RESIDUALS ** Opt over Cxyz <<<<<<< ")
    print("Initials Cxyz:{}".format(initial_Cxyz))
    print("Opt Cxy:{}".format(opt_Cxyz))
    print("Iterations: {}".format(info[2]))
    print("termination: {}".format(info[3]))
    residuals = residuals_error_Cxyz(
        parameters=opt_Cxyz,
        bearings_kf=kwargs["bearings"]["kf"].copy(),
        bearings_frm=kwargs["bearings"]["frm"].copy(),
    )

    bearings_kf_norm, t1 = normalizer_Cxyz(
        x=kwargs["bearings"]["kf"].copy(),
        Cxyz=opt_Cxyz)

    bearings_frm_norm, t2 = normalizer_Cxyz(
        x=kwargs["bearings"]["frm"].copy(),
        Cxyz=opt_Cxyz)

    e_norm = solver.compute_essential_matrix(
        x1=bearings_kf_norm,
        x2=bearings_frm_norm
    )
    e_hat = t2.T @ e_norm @ t1

    cam_final = solver.recover_pose_from_e(
        E=e_hat,
        x1=kwargs["bearings"]['kf'].copy(),
        x2=kwargs["bearings"]['frm'].copy(),
    )

    return cam_final, np.sum(residuals ** 2)
