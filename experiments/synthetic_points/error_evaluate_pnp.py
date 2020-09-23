from pcl_utilities import *

from read_datasets.MP3D_VO import MP3D_VO
from read_datasets.TUM_RGBD import TUM_RGBD

from geometry_utilities import *
from config import *


def eval_error(res, noise, loc, point, data_scene, idx_frame, opt_version,
               scene, motion_constraint):
    # ! relative camera pose from a to b
    error_n8p, error_8p = [], []

    from solvers.pnp import PnP_DLT

    pnp_dlt = PnP_DLT()

    # ! Getting a PCL from the dataset
    pcl_dense, pcl_dense_color, _, _ = data_scene.get_pcl(idx=idx_frame)
    pcl_dense, mask = mask_pcl_by_res_and_loc(pcl=pcl_dense, loc=loc, res=res)
    np.random.seed(100)

    for _ in range(100):
        # ! relative camera pose from a to b
        cam_gt = get_homogeneous_transform_from_vectors(
            t_vector=(np.random.uniform(-1, 1), np.random.uniform(-0.5, 0.5),
                      np.random.uniform(-1, 1)),
            r_vector=(np.random.uniform(-10, 10), np.random.uniform(-10, 10),
                      np.random.uniform(-10, 10)))

        # cam_a2b = get_homogeneous_transform_from_vectors(t_vector=(0, 1, 0),
        #                                                  r_vector=(0, 30, 0))

        samples = np.random.randint(0, pcl_dense.shape[1], point)
        pcl_a = extend_array_to_homogeneous(pcl_dense[:, samples])
        # ! pcl at "b" location + noise
        pcl_b = np.linalg.inv(cam_gt).dot(pcl_a)
        # pcl_b = add_noise_to_pcl(np.linalg.inv(cam_a2b).dot(pcl_a),
        #                          param=noise)

        # ! We expect that there are 1% outliers besides of the noise
        # pcl_b = add_outliers_to_pcl(pcl_b.copy(), outliers=int(0.05 * point))

        bearings_a = sph.sphere_normalization(pcl_a)
        bearings_b = sph.sphere_normalization(pcl_b)

        cam_pnp = pnp_dlt.recoverPose(w=pcl_a.copy(), x=bearings_b.copy())

        error_8p.append(
            evaluate_error_in_transformation(
                transform_gt=cam_gt, transform_est=cam_pnp))

        print(
            "====================================================================="
        )
        # ! PnP
        print("Q1-PnP: {} - {}".format(
            np.quantile(error_8p, 0.25, axis=0), len(error_8p)))
        print("Q2-PnP: {} - {}".format(
            np.median(error_8p, axis=0), len(error_8p)))
        print("Q3-PnP: {} - {}".format(
            np.quantile(error_8p, 0.75, axis=0), len(error_8p)))
        print(
            "====================================================================="
        )


if __name__ == '__main__':
    assert experiment == experiment_choices[0]

    if dataset == "minos":
        data = MP3D_VO(basedir=basedir, scene=scene)

    if experiment_group == "noise":
        for noise in noises:
            eval_error(
                res=res,
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
            eval_error(
                res=res,
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
            eval_error(
                res=res,
                noise=noise,
                loc=(0, 0),
                point=point,
                data_scene=data,
                idx_frame=idx_frame,
                opt_version=opt_version,
                scene=scene,
                motion_constraint=motion_constraint)
