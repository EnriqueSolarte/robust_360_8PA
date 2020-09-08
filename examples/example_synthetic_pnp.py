from pcl_utilities import *
from geometry_utilities import *


def evaluate_synthetic_points(theta_roi, phi_roi, n_pts, min_d, max_d,
                              relative_cam_pose):
    # ! relative camera pose from a to b
    cam_gt = relative_cam_pose
    error_8p = []

    from solvers.pnp import PnP
    pnp = PnP()

    for _ in range(10):
        pcl_a = generate_pcl_by_roi_theta_phi(theta=theta_roi,
                                              phi=phi_roi,
                                              n_pts=n_pts,
                                              min_d=min_d,
                                              max_d=max_d)

        pcl_b = np.linalg.inv(cam_gt).dot(pcl_a)
        bearings_b = sph.sphere_normalization(pcl_b)

        cam_pnp = pnp.recoverPose(
            w=pcl_a.copy(),
            x=bearings_b.copy())

        error_8p.append(
            evaluate_error_in_transformation(transform_gt=cam_gt,
                                             transform_est=cam_pnp))

        # ! 8PA
        print("Q1-8PA:{} - {}".format(np.quantile(error_8p, 0.25, axis=0),
                                      len(error_8p)))
        print("Q2-8PA:{} - {}".format(np.median(error_8p, axis=0),
                                      len(error_8p)))
        print("Q3-8PA:{} - {}".format(np.quantile(error_8p, 0.75, axis=0),
                                      len(error_8p)))
        print(
            "====================================================================="
        )


if __name__ == '__main__':
    # ! relative camera pose from a to b
    cam_pose = get_homogeneous_transform_from_vectors(
        t_vector=(np.random.uniform(-1, 1), np.random.uniform(-0.5, 0.5),
                  np.random.uniform(-1, 1)),
        r_vector=(np.random.uniform(-10, 10), np.random.uniform(-10, 10),
                  np.random.uniform(-10, 10)))

    delta_theta = -0
    delta_phi = -0
    cfg = dict(theta_roi=(-180 + delta_theta, 180 + delta_theta),
               phi_roi=(-90 + delta_phi, 90 + delta_phi),
               n_pts=100,
               min_d=2,
               max_d=20,
               relative_cam_pose=cam_pose,
               )
    evaluate_synthetic_points(**cfg)
