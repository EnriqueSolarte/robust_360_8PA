from solvers.epipolar_constraint import EightPointAlgorithmGeneralGeometry
from solvers.optimal8pa import Optimal8PA as norm_8pa
from pcl_utilities import *
from read_datasets.MP3D_VO import MP3D_VO
from geometry_utilities import *
from file_utilities import FileReport
from structures.extractor.orb_extractor import ORBExtractor
from structures.extractor.shi_tomasi_extractor import Shi_Tomasi_Extractor


def eval_methods(res, noise, loc, data_scene, idx_frame, opt_version, feat_extractor):
    # ! relative camera pose from a to b
    error_n8p = []
    error_8p = []

    g8p_norm = norm_8pa(version=opt_version)
    g8p = EightPointAlgorithmGeneralGeometry()

    # ! Getting a PCL from the dataset
    pcl = data_scene.get_pcl_from_key_features(idx=idx_frame, extractor=feat_extractor)
    # pcl_dense, pcl_dense_color, _ = data_scene.get_dense_pcl(idx=idx_frame)
    pcl, mask = mask_pcl_by_res_and_loc(pcl=pcl, loc=loc, res=res)
    np.random.seed(100)

    while True:
        # ! relative camera pose from a to b
        cam_a2b = get_homogeneous_transform_from_vectors(
            t_vector=(np.random.uniform(-1, 1), np.random.uniform(-1, 1),
                      np.random.uniform(-1, 1)),
            r_vector=(np.random.uniform(-10, 10), np.random.uniform(-10, 10),
                      np.random.uniform(-10, 10)))

        pcl_a = extend_array_to_homogeneous(pcl)
        # ! pcl at "b" location + noise
        pcl_b = add_noise_to_pcl(np.linalg.inv(cam_a2b).dot(pcl_a),
                                 param=noise)
        # ! We expect that there are 1% outliers besides of the noise
        pcl_b = add_outliers_to_pcl(pcl_b.copy(), outliers=int(0.05 * pcl_a.shape[1]))
        bearings_a = sph.sphere_normalization(pcl_a)
        bearings_b = sph.sphere_normalization(pcl_b)

        cam_a2b_8p = g8p.recover_pose_from_matches(x1=bearings_a.copy(),
                                                   x2=bearings_b.copy())

        cam_a2b_n8p = g8p_norm.recover_pose_from_matches(x1=bearings_a.copy(),
                                                         x2=bearings_b.copy())

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


if __name__ == '__main__':
    # scene = "1LXtFkjw3qL/1"
    scene = "2azQ1b91cZZ/0"
    path = "/home/kike/Documents/datasets/MP3D_VO"
    data = MP3D_VO(scene=scene, path=path)

    eval_methods(res=(54, 54),
                 noise=500,
                 loc=(0, 0),
                 feat_extractor=ORBExtractor(),
                 # feat_extractor = Shi_Tomasi_Extractor(),
                 data_scene=data,
                 idx_frame=150,
                 opt_version="v2")
