from Solvers.EpipolarConstraint import EightPointAlgorithmGeneralGeometry
from Solvers.Optimal8PA import Optimal8PA
from pcl_utilities import *
from geometry_utilities import *


def evaluate_synthetic_points(theta_roi, phi_roi,
                              n_pts, min_d, max_d,
                              relative_cam_pose,
                              noise_parameter):
    # ! relative camera pose from a to b
    cam_a2b = relative_cam_pose
    error_n8p = []
    error_8p = []
    g8p_norm = Optimal8PA()
    g8p = EightPointAlgorithmGeneralGeometry()

    while True:
        pcl_a = generate_pcl_by_roi_theta_phi(theta=theta_roi,
                                              phi=phi_roi,
                                              n_pts=n_pts,
                                              min_d=min_d,
                                              max_d=max_d)
        # ! pcl at "b" location + noise
        pcl_b = add_noise_to_pcl(np.linalg.inv(cam_a2b).dot(pcl_a), param=noise_parameter)
        pcl_b = add_outliers_to_pcl(pcl_b.copy(), outliers=5)
        bearings_a = sph.sphere_normalization(pcl_a)
        bearings_b = sph.sphere_normalization(pcl_b)

        samples = np.random.randint(0, pcl_b.shape[1], n_pts)
        x1 = bearings_a[:, samples]
        x2 = bearings_b[:, samples]

        cam_a2b_n8p = g8p_norm.recover_pose_from_matches(x1=x1.copy(), x2=x2.copy())
        cam_a2b_8p = g8p.recover_pose_from_matches(x1=x1.copy(), x2=x2.copy())

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
        print("ours:{} - {}".format(np.median(error_n8p, axis=0), len(error_n8p)))
        print("8PA: {} - {}".format(np.median(error_8p, axis=0), len(error_8p)))


if __name__ == '__main__':
    # ! relative camera pose from a to b
    cam_pose = get_homogeneous_transform_from_vectors(t_vector=(0, 0, 0.5),
                                                      r_vector=(np.random.uniform(-10, 10),
                                                                np.random.uniform(-10, 10),
                                                                np.random.uniform(-10, 10)))
    delta_theta = -0
    delta_phi = -0
    cfg = dict(theta_roi=(-27 + delta_theta, 27 + delta_theta),
               phi_roi=(-18 + delta_phi, 18 + delta_phi),
               n_pts=200, min_d=2, max_d=20,
               relative_cam_pose=cam_pose,
               noise_parameter=500)
    evaluate_synthetic_points(**cfg)
