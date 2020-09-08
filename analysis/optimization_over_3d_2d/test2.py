from read_datasets.MP3D_VO import MP3D_VO
from structures.tracker import LKTracker
from structures.extractor.shi_tomasi_extractor import Shi_Tomasi_Extractor
from utilities.data_utilities import *
from utilities.optmization_utilities import *
import levmar
import time

solver = g8p()


def eval_normalizer(parameters, bearings_kf, bearings_frm):
    s1 = parameters[0]
    k1 = parameters[1]
    s2 = parameters[2]
    k2 = parameters[3]

    # bearings_kf_norm, t1 = normalizer_s_k(x=bearings_kf, s=s1, k=k1)
    # bearings_frm_norm, t2 = normalizer_s_k(x=bearings_frm, s=s2, k=k2)
    bearings_kf_norm, t1 = normalizer_s(x=bearings_kf, s=s1)
    bearings_frm_norm, t2 = normalizer_s(x=bearings_frm, s=s2)

    e_norm = solver.compute_essential_matrix(
        x1=bearings_kf_norm,
        x2=bearings_frm_norm
    )
    e_hat = t1.T @ e_norm @ t2

    cam_hat = solver.recover_pose_from_e(
        E=e_hat,
        x1=bearings_kf,
        x2=bearings_frm
    )
    return cam_hat


def reprojection_error_R_T(parameters, bearings, points):
    r0 = parameters[0]
    r1 = parameters[1]
    r2 = parameters[2]
    t0 = parameters[3]
    t1 = parameters[4]
    t2 = parameters[5]

    cam_pose = eulerAnglesToRotationMatrix((r0, r1, r2))
    cam_pose[0:3, 3] = np.array((t0, t1, t2))

    points_hat = np.linalg.inv(cam_pose) @ points
    error = get_angle_between_vectors_arrays(
        array_ref=bearings,
        array_vector=points_hat[0:3, :]
    )
    return error


def reprojection_error_S_K(parameters, bearings_kf, bearings_frm):
    cam_hat = eval_normalizer(parameters, bearings_kf, bearings_frm)
    landmarks_kf_hat = g8p.triangulate_points_from_cam_pose(
        cam_pose=cam_hat,
        x1=bearings_kf,
        x2=bearings_frm
    )
    landmarks_frm_hat = np.linalg.inv(cam_hat) @ landmarks_kf_hat
    error = get_angle_between_vectors_arrays(
        array_ref=bearings_frm,
        array_vector=landmarks_frm_hat[0:3, :]
    )
    return error


def reprojection_error_S_K_const_lm(parameters, bearings_kf, bearings_frm, landmarks_kf_hat):
    cam_hat = eval_normalizer(parameters, bearings_kf, bearings_frm)
    landmarks_frm_hat = np.linalg.inv(cam_hat) @ landmarks_kf_hat
    error = get_angle_between_vectors_arrays(
        array_ref=bearings_frm,
        array_vector=landmarks_frm_hat[0:3, :]
    )
    return error


def run_estimation(**kwargs):
    kwargs = get_bearings(**kwargs)

    # ! Getting initial data
    tic = time.time()
    cam_8p = g8p().recover_pose_from_matches(
        x1=kwargs["bearings"]["kf"],
        x2=kwargs["bearings"]["frm"]
    )
    landmarks_8p_kf = g8p.triangulate_points_from_cam_pose(
        cam_pose=cam_8p,
        x1=kwargs["bearings"]["kf"],
        x2=kwargs["bearings"]["frm"]
    )
    time_8p = time.time() - tic
    print("opt 8p: {}".format(time_8p))

    eu = rotationMatrixToEulerAngles(cam_8p[0:3, 0:3])
    trn = np.copy(cam_8p[0:3, 3])
    initial_R_t = np.hstack((eu, trn))
    tic = time.time()
    opt_R_t, p_cov, info = levmar.levmar(
        reprojection_error_R_T,
        initial_R_t,
        np.ones_like(kwargs["bearings"]["frm"][0, :]),
        args=(kwargs["bearings"]["frm"], landmarks_8p_kf))
    time_rt = time.time() - tic + time_8p
    print("opt R, t: {}".format(time_rt))

    initial_k_s = np.array((1, 1, 1, 1))
    # tic = time.time()
    # _ = g8p.triangulate_points_from_cam_pose(
    #     cam_pose=cam_8p,
    #     x1=kwargs["bearings"]["kf"],
    #     x2=kwargs["bearings"]["frm"]
    # )
    # opt_k_s, p_cov, info = levmar.levmar(
    #     reprojection_error_S_K_const_lm,
    #     initial_k_s,
    #     np.zeros_like(kwargs["bearings"]["frm"][0, :]),
    #     args=(kwargs["bearings"]["kf"],
    #           kwargs["bearings"]["frm"],
    #           landmarks_8p_kf))
    # opt_k_s, p_cov, info = levmar.levmar(
    #     reprojection_error_S_K,
    #     initial_k_s,
    #     np.zeros_like(kwargs["bearings"]["frm"][0, :]),
    #     args=(kwargs["bearings"]["kf"],
    #           kwargs["bearings"]["frm"]))
    toc = time.time()
    print("opt S, K: {}".format(toc - tic))

    print("Initial projection error (8p): {}".format(np.sum(reprojection_error_S_K(
        initial_k_s,
        kwargs["bearings"]["kf"],
        kwargs["bearings"]["frm"])
    )))

    # print("Final projection error by Opt(s,k) : {}".format(np.sum(reprojection_error_S_K(
    #     opt_k_s,
    #     kwargs["bearings"]["kf"],
    #     kwargs["bearings"]["frm"])
    # )))

    print("Final projection error by Opt(R,t): {}".format(np.sum(reprojection_error_R_T(
        opt_R_t,
        bearings=kwargs["bearings"]["frm"],
        points=landmarks_8p_kf
    ))))

    print("Initial camera error (8p): {}".format(
        evaluate_error_in_transformation(
            transform_gt=kwargs["cam_gt"],
            transform_est=cam_8p
        )
    ))

    # print("Final camera error by Opt(s,k): {}".format(
    #     evaluate_error_in_transformation(
    #         transform_gt=kwargs["cam_gt"],
    #         transform_est=eval_normalizer(
    #             parameters=opt_k_s,
    #             bearings_kf=kwargs["bearings"]["kf"],
    #             bearings_frm=kwargs["bearings"]["frm"],
    #         )
    #     )
    # ))

    cam_final = eulerAnglesToRotationMatrix(opt_R_t[0:3])
    cam_final[0:3, 3] = opt_R_t[3:]
    print("Final camera error by Opt(R,t): {}".format(
        evaluate_error_in_transformation(
            transform_gt=kwargs["cam_gt"],
            transform_est=cam_final
        )
    ))


if __name__ == '__main__':
    path = "/home/kike/Documents/datasets/MP3D_VO"
    # scene = "2azQ1b91cZZ/0"
    scene = "1LXtFkjw3qL/0"
    # scene = "759xd9YjKW5/0"
    # path = "/run/user/1001/gvfs/sftp:host=140.114.27.95,port=50002/NFS/kike/minos/vslab_MP3D_VO/512x1024"
    data = MP3D_VO(scene=scene, basedir=path)

    scene_settings = dict(
        data_scene=data,
        idx_frame=0,
        distance_threshold=0.5,
        # res=(360, 180),
        res=(180, 180),
        # res=(65.5, 46.4),
        loc=(0, 0),
    )

    model_settings = dict(
        opt_version="v1.1",
        # extra="epipolar_constraint"
        extra="projected_distance",
        # extra="sampson_distance",
        # extra="tangential_distance"
    )

    ransac_parm = dict(min_samples=8,
                       max_trials=RansacEssentialMatrix.get_number_of_iteration(
                           p_success=0.99, outliers=0.5, min_constraint=8
                       ),
                       residual_threshold=5e-5,
                       verbose=True,
                       use_ransac=True
                       )

    features_setting = dict(
        feat_extractor=Shi_Tomasi_Extractor(),
        tracker=LKTracker(),
        show_tracked_features=True
    )

    run_estimation(**scene_settings,
                   **features_setting,
                   **ransac_parm,
                   **model_settings)
