from solvers.epipolar_constraint import EightPointAlgorithmGeneralGeometry
from solvers.optimal8pa import Optimal8PA as norm_8pa
from pcl_utilities import *
from read_datasets.MP3D_VO import MP3D_VO
from geometry_utilities import *
from file_utilities import FileReport, create_dir, create_file


def eval_methods(res, noise, loc, pts, data_scene, idx_frame, opt_version,
                 scene, motion_constraint):
    # ! relative camera pose from a to b
    error_n8p = []
    error_8p = []

    g8p_norm = norm_8pa(version=opt_version)
    g8p = EightPointAlgorithmGeneralGeometry()

    # ! Getting a PCL from the dataset
    pcl_dense, pcl_dense_color, _ = data_scene.get_dense_pcl(idx=idx_frame)
    pcl_dense, mask = mask_pcl_by_roi_and_loc(pcl=pcl_dense, loc=loc, res=res)
    np.random.seed(100)

    # ! Output file
    # filename = "../../report/{}_{}_{}x{}_sample_scene.csv".format(opt_version, scene, str(res[0]), str(res[1]))
    error_report = FileReport(
        filename="../../report/{}/{}/{}/{}/{}_{}x{}.csv".format(
            scene, str(idx_frame), "mc" if motion_constraint else "~mc", noise,
            opt_version, str(res[0]), str(res[1])))
    error_report.set_headers(["rot-8PA", "tran-8PA", "rot-n8PA", "tran-n8PA"])

    for _ in range(500):
        # ! relative camera pose from a to b
        cam_a2b = get_homogeneous_transform_from_vectors(
            t_vector=(np.random.uniform(-1, 1), np.random.uniform(-1, 1),
                      np.random.uniform(-1, 1)),
            r_vector=(np.random.uniform(-10, 10), np.random.uniform(-10, 10),
                      np.random.uniform(-10, 10)))

        # cam_a2b = get_homogeneous_transform_from_vectors(t_vector=(0, 1, 0),
        #                                                  r_vector=(0, 0, 0))

        samples = np.random.randint(0, pcl_dense.shape[1], pts)
        pcl_a = extend_array_to_homogeneous(pcl_dense[:, samples])
        # ! pcl at "b" location + noise
        pcl_b = add_noise_to_pcl(np.linalg.inv(cam_a2b).dot(pcl_a),
                                 param=noise)
        # ! We expect that there are 1% outliers besides of the noise
        # pcl_b = add_outliers_to_pcl(pcl_b.copy(), outliers=int(0.05 * pts))
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
            #
            cam_a2b_n8p_rot = g8p_norm.recover_pose_from_matches(
                x1=bearings_a_rot.copy(), x2=bearings_b_rot.copy())

            cam_a2b_n8p = extend_SO3_to_homogenous(
                rot.T).dot(cam_a2b_n8p_rot).dot(extend_SO3_to_homogenous(rot))
        else:
            cam_a2b_n8p = g8p_norm.recover_pose_from_matches(
                x1=bearings_a.copy(), x2=bearings_b.copy())

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
            error_n8p[-1][1]
        ]
        error_report.write(line)


if __name__ == '__main__':
    # path = "/home/kike/Documents/datasets/Matterport_360_odometry"
    path = "/run/user/1001/gvfs/sftp:host=140.114.27.95,port=50002/NFS/kike/minos/vslab_MP3D_VO/512x1024"
    scene = "1LXtFkjw3qL"
    data = MP3D_VO(scene=scene + "/0", path=path)
    idx_frame = 0
    motion_constraint = False

    ress = [(54.4, 37.8), (65.5, 46.4), (195, 195), (360, 180)]
    # ress = [(3.44, 5.15), (16.1, 23.9), (27.0, 39.6), (54.4, 37.8), (65.5, 46.4), (81.2, 102.7), (195, 195), (360, 180)]
    noises = [500, 1000, 2000, 10000]

    create_dir("../../report/{}/{}/{}".format(
        scene, str(idx_frame), "mc" if motion_constraint else "~mc"),
               delete_previous=False)

    for noise in noises:
        create_dir("../../report/{}/{}/{}/{}".format(
            scene, str(idx_frame), "mc" if motion_constraint else "~mc",
            noise),
                   delete_previous=False)
        eval_methods(res=ress[1],
                     noise=noise,
                     loc=(0, 0),
                     pts=150,
                     data_scene=data,
                     idx_frame=idx_frame,
                     opt_version="v2",
                     scene=scene,
                     motion_constraint=motion_constraint)
