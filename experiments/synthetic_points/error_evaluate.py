from pcl_utilities import *

from read_datasets.MP3D_VO import MP3D_VO
from read_datasets.TUM_RGBD import TUM_RGBD

from geometry_utilities import *
from file_utilities import FileReport, create_dir, create_file
from config import *


def eval_error(res, noise, loc, point, data_scene, idx_frame, opt_version,
               scene, motion_constraint):
    # ! relative camera pose from a to b
    error_n8p, error_8p = [], []
    s1, s2, k1, k2 = 0, 0, 0, 0

    from solvers.epipolar_constraint import EightPointAlgorithmGeneralGeometry as g8p
    from solvers.optimal8pa import Optimal8PA as norm_8pa

    g8p_norm = norm_8pa(version=opt_version)
    g8p = g8p()

    # ! Getting a PCL from the dataset
    pcl_dense, pcl_dense_color, _ = data_scene.get_pcl(idx=idx_frame)
    pcl_dense, mask = mask_pcl_by_res_and_loc(pcl=pcl_dense, loc=loc, res=res)
    np.random.seed(100)

    # ! Output file
    filename = "../../report/{}/{}/{}/{}/{}/{}/{}/{}/{}_{}_{}_{}_{}_{}_{}_{}.csv".format(
        experiment, dataset, scene, str(idx_frame),
        "mc" if motion_constraint else "!mc", noise,
        str(res[0]) + "x" + str(res[1]), point, scene[:-2], scene[-1:],
        str(idx_frame), "mc" if motion_constraint else "!mc", noise,
        str(res[0]) + "x" + str(res[1]), point, opt_version)

    error_report = FileReport(filename=filename)
    error_report.set_headers([
        "rot-8PA", "tran-8PA", "rot-n8PA", "tran-n8PA", "s1", "k1", "s2", "k2"
    ])

    for _ in range(100):
        # ! relative camera pose from a to b
        cam_a2b = get_homogeneous_transform_from_vectors(
            t_vector=(np.random.uniform(-1, 1), np.random.uniform(-1, 1),
                      np.random.uniform(-1, 1)),
            r_vector=(np.random.uniform(-10, 10), np.random.uniform(-10, 10),
                      np.random.uniform(-10, 10)))

        # cam_a2b = get_homogeneous_transform_from_vectors(t_vector=(0, 1, 0),
        #                                                  r_vector=(0, 0, 0))

        samples = np.random.randint(0, pcl_dense.shape[1], point)
        pcl_a = extend_array_to_homogeneous(pcl_dense[:, samples])
        # ! pcl at "b" location + noise
        pcl_b = add_noise_to_pcl(np.linalg.inv(cam_a2b).dot(pcl_a),
                                 param=noise)

        # ! We expect that there are 1% outliers besides of the noise
        # pcl_b = add_outliers_to_pcl(pcl_b.copy(), outliers=int(0.05 * point))

        bearings_a = sph.sphere_normalization(pcl_a)
        bearings_b = sph.sphere_normalization(pcl_b)

        cam_a2b_8p = g8p.recover_pose_from_matches(x1=bearings_a.copy(),
                                                   x2=bearings_b.copy())

        if motion_constraint:
            # # ! prior motion
            prior_motion = cam_a2b_8p[0:3, 3]
            # prior_motion = cam_a2b[0:3, 3]

            rot = get_rot_from_directional_vectors(prior_motion, (0, 0, 1))
            bearings_a_rot = rot.dot(bearings_a)
            bearings_b_rot = rot.dot(bearings_b)

            cam_a2b_n8p_rot = g8p_norm.recover_pose_from_matches(
                x1=bearings_a_rot.copy(), x2=bearings_b_rot.copy())

            cam_a2b_n8p = extend_SO3_to_homogenous(
                rot.T).dot(cam_a2b_n8p_rot).dot(extend_SO3_to_homogenous(rot))
        else:
            cam_a2b_n8p = g8p_norm.recover_pose_from_matches(
                x1=bearings_a.copy(), x2=bearings_b.copy())

            s1 = g8p_norm.T1[0][0]
            k1 = g8p_norm.T1[2][2]
            print("s1, k1 = ({}, {})".format(s1, k1))

            if opt_version != "v1":
                s2 = g8p_norm.T2[0][0]
                k2 = g8p_norm.T2[2][2]
                print("s2, k2 = ({}, {})".format(s2, k2))

        if cam_a2b_8p is None:
            print("8p failed")
            continue
        if cam_a2b_n8p is None:
            print("n8p failed")
            continue

        error_n8p.append(
            evaluate_error_in_transformation(transform_gt=cam_a2b,
                                             transform_est=cam_a2b_n8p))
        error_8p.append(
            evaluate_error_in_transformation(transform_gt=cam_a2b,
                                             transform_est=cam_a2b_8p))

        print(
            "====================================================================="
        )
        # ! Ours' method
        print("Q1-ours:{}- {}".format(np.quantile(error_n8p, 0.25, axis=0),
                                      len(error_n8p)))
        print("Q2-ours:{}- {}".format(np.median(error_n8p, axis=0),
                                      len(error_n8p)))
        print("Q3-ours:{}- {}".format(np.quantile(error_n8p, 0.75, axis=0),
                                      len(error_n8p)))

        print(
            "====================================================================="
        )
        # ! 8PA
        print("Q1-8PA:{}-  {}".format(np.quantile(error_8p, 0.25, axis=0),
                                      len(error_8p)))
        print("Q2-8PA:{}-  {}".format(np.median(error_8p, axis=0),
                                      len(error_8p)))
        print("Q3-8PA:{}-  {}".format(np.quantile(error_8p, 0.75, axis=0),
                                      len(error_8p)))
        print(
            "====================================================================="
        )

        line = [
            error_8p[-1][0], error_8p[-1][1], error_n8p[-1][0],
            error_n8p[-1][1], s1, k1, s2, k2
        ]
        error_report.write(line)


if __name__ == '__main__':
    assert experiment == experiment_choices[0]

    if dataset == "minos":
        data = MP3D_VO(basedir=path, scene=scene)
    # elif dataset == "tum_rgbd":
    #     data = undistort_depth(path=path, scene=scene)

    if experiment_group == "noise":
        for noise in noises:
            create_dir("../../report/{}/{}/{}/{}/{}/{}/{}/{}".format(
                experiment, dataset, scene, str(idx_frame),
                "mc" if motion_constraint else "!mc", noise,
                str(res[0]) + "x" + str(res[1]), point),
                       delete_previous=False)
            eval_error(res=res,
                       noise=noise,
                       loc=(0, 0),
                       point=point,
                       data_scene=data,
                       idx_frame=idx_frame,
                       opt_version=opt_version,
                       scene=scene,
                       motion_constraint=motion_constraint)
    elif experiment_group == "fov":
        for res in ress:
            create_dir("../../report/{}/{}/{}/{}/{}/{}/{}/{}".format(
                experiment, dataset, scene, str(idx_frame),
                "mc" if motion_constraint else "!mc", noise,
                str(res[0]) + "x" + str(res[1]), point),
                       delete_previous=False)
            eval_error(res=res,
                       noise=noise,
                       loc=(0, 0),
                       point=point,
                       data_scene=data,
                       idx_frame=idx_frame,
                       opt_version=opt_version,
                       scene=scene,
                       motion_constraint=motion_constraint)
    elif experiment_group == "point":
        for point in points:
            create_dir("../../report/{}/{}/{}/{}/{}/{}/{}/{}".format(
                experiment, dataset, scene, str(idx_frame),
                "mc" if motion_constraint else "!mc", noise,
                str(res[0]) + "x" + str(res[1]), point),
                       delete_previous=False)
            eval_error(res=res,
                       noise=noise,
                       loc=(0, 0),
                       point=point,
                       data_scene=data,
                       idx_frame=idx_frame,
                       opt_version=opt_version,
                       scene=scene,
                       motion_constraint=motion_constraint)
