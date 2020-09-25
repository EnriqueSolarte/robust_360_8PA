import numpy as np
from vslam360_src.Ransac.RansacEssentialMatrix import RansacEssentialMatrix
from pcl_utilities import generate_pcl_by_roi_theta_phi, add_noise_to_pcl
from geometry_utilities import get_homogeneous_transform_from_vectors, evaluate_error_in_transformation
from Sphere import Sphere as sph


def evaluate_synthetic_points(theta_roi, phi_roi, n_pts, min_d, max_d,
                              relative_cam_pose, noise, outliers, threshold):

    ransac_parm = dict(
        min_samples=8,
        p_succes=0.99,
        outliers=outliers,  # * expecting 50% of outliers
        residual_threshold=threshold,
        verbose=True)

    # ! relative camera pose from a to b
    cam_a2b = relative_cam_pose
    error_n8p = []
    error_8p = []
    for trial in range(100):

        # ! pcl at "a" location
        pcl_a = generate_pcl_by_roi_theta_phi(theta=theta_roi,
                                              phi=phi_roi,
                                              n_pts=n_pts,
                                              min_d=min_d,
                                              max_d=max_d)

        # ! pcl at "b" location
        pcl_b = add_noise_to_pcl(np.linalg.inv(cam_a2b).dot(pcl_a),
                                 param=noise)
        #                        , outlier_ratio = outliers)

        cam_a2b_n8p = RansacEssentialMatrix(**ransac_parm).solve(
            data=(sph.sphere_normalization(pcl_a).T,
                  sph.sphere_normalization(pcl_b).T),
            solver="norm_8pa")

        cam_a2b_8p = RansacEssentialMatrix(**ransac_parm).solve(
            data=(sph.sphere_normalization(pcl_a).T,
                  sph.sphere_normalization(pcl_b).T),
            solver="g8p")

        error_n8p.append(
            evaluate_error_in_transformation(transform_gt=cam_a2b,
                                             transform_est=cam_a2b_n8p))

        error_8p.append(
            evaluate_error_in_transformation(transform_gt=cam_a2b,
                                             transform_est=cam_a2b_8p))

    print(theta_roi, phi_roi)
    print("Error n8PA: {}".format(np.median(np.array(error_n8p), axis=0)))
    print("Error  8PA: {}".format(np.median(np.array(error_8p), axis=0)))


if __name__ == '__main__':
    # ! relative camera pose from a to b
    cam_pose = get_homogeneous_transform_from_vectors(t_vector=(0, 0, 0.5),
                                                      r_vector=(10, 10, 10))

    from config import *
    fov = (90, 90)
    cfg = dict(theta_roi=(-fov[1] / 2, fov[1] / 2),
               phi_roi=(-fov[0] / 2, fov[0] / 2),
               n_pts=200,
               min_d=2,
               max_d=20,
               relative_cam_pose=cam_pose,
               noise=10,
               outliers=0.5,
               threshold=1)
    evaluate_synthetic_points(**cfg)
