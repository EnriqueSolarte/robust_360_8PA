from solvers.epipolar_constraint import EightPointAlgorithmGeneralGeometry
from solvers.optimal8pa import Optimal8PA as norm_8pa
from pcl_utilities import *
from file_utilities import create_dir, create_file, write_report


def noise_evaluation(res, noise, loc, pts, data_scene, output_dir):
    # ! relative camera pose from a to b
    error_n8p = []
    error_8p = []

    g8p_norm = norm_8pa()
    g8p = EightPointAlgorithmGeneralGeometry()

    # ! Gebereating PCL from the dataset
    pcl_dense, _ = get_dense_pcl_sample(scene=data_scene["scene"],
                                        path=data_scene["path"],
                                        idx=data_scene["idx"],
                                        res=res, loc=loc)
    np.random.seed(100)

    # ! Output file

    while True:
        # ! relative camera pose from a to b
        cam_a2b = get_homogeneous_transform_from_vectors(t_vector=(np.random.uniform(-1, 1),
                                                                   np.random.uniform(-1, 1),
                                                                   np.random.uniform(-1, 1)),
                                                         r_vector=(np.random.uniform(-10, 10),
                                                                   np.random.uniform(-10, 10),
                                                                   np.random.uniform(-10, 10)))

        samples = np.random.randint(0, pcl_dense.shape[1], pts)
        pcl_a = pcl_dense[:, samples]
        # ! pcl at "b" location + noise
        pcl_b = add_noise_to_pcl(np.linalg.inv(cam_a2b).dot(pcl_a), param=noise)
        # ! We expect that there are 1% outliers besides of the noise
        pcl_b = add_outliers_to_pcl(pcl_b.copy(), outliers=int(0.05 * pts))
        bearings_a = sph.sphere_normalization(pcl_a)
        bearings_b = sph.sphere_normalization(pcl_b)

        cam_a2b_8p = g8p.recover_pose_from_matches(x1=bearings_a.copy(), x2=bearings_b.copy())
        cam_a2b_n8p = g8p_norm.recover_pose_from_matches(x1=bearings_a.copy(), x2=bearings_b.copy())

        if cam_a2b_8p is None:
            print("8p failed")
            continue
        if cam_a2b_n8p is None:
            print("n8p failed")
            continue

        error_n8p.append(evaluate_error_in_transformation(transform_gt=cam_a2b,
                                                          transform_est=cam_a2b_n8p))
        error_8p.append(evaluate_error_in_transformation(transform_gt=cam_a2b,
                                                         transform_est=cam_a2b_8p))

        print("=====================================================================")
        # ! Ours' method
        print("Q1-ours:{}- {}".format(np.quantile(error_n8p, 0.25, axis=0),
                                      len(error_n8p)))
        print("Q2-ours:{}- {}".format(np.median(error_n8p, axis=0),
                                      len(error_n8p)))
        print("Q3-ours:{}- {}".format(np.quantile(error_n8p, 0.75, axis=0),
                                      len(error_n8p)))

        print("=====================================================================")
        # ! 8PA
        print("Q1-8PA:{}-  {}".format(np.quantile(error_8p, 0.25, axis=0),
                                      len(error_8p)))
        print("Q2-8PA:{}-  {}".format(np.median(error_8p, axis=0),
                                      len(error_8p)))
        print("Q3-8PA:{}-  {}".format(np.quantile(error_8p, 0.75, axis=0),
                                      len(error_8p)))


if __name__ == '__main__':
    scene = dict(scene="1LXtFkjw3qL",
                 path="0",
                 idx=0)

    noise_evaluation(res=fov, noise=noise, loc=(-0, 0), pts=200, data_scene=scene, output_dir=output_dir)
