from delta_bound import *

# pcl_main = generate_pcl_by_roi_theta_phi(theta=(-180, 180),
#                                          phi=(-90, 90),
#                                          n_pts=100,
#                                          min_d=2,
#                                          max_d=20)

scene = dict(scene="1LXtFkjw3qL",
             path="0",
             idx=50)
pcl_main, _ = get_dense_pcl_sample(scene=scene["scene"],
                                   path=scene["path"],
                                   idx=scene["idx"],
                                   res=(360, 180), loc=(0, 0))

solver = ng8p()
for trial in range(1):
    cam_a2b = get_homogeneous_transform_from_vectors(t_vector=(np.random.uniform(-1, 1),
                                                               np.random.uniform(-1, 1),
                                                               np.random.uniform(-1, 1)),
                                                     r_vector=(np.random.uniform(-10, 10),
                                                               np.random.uniform(-10, 10),
                                                               np.random.uniform(-10, 10)))

    samples = np.random.randint(0, pcl_main.shape[1], 200)
    pcl = add_noise_to_pcl(np.linalg.inv(cam_a2b).dot(pcl_main[:, samples]), param=500)
    bearings_a = sph.sphere_normalization(pcl)
    bearings_b = sph.sphere_normalization(np.linalg.inv(cam_a2b).dot(pcl))

    prior_motion = cam_a2b[0:3, 3]
    rot = get_rot_from_directional_vectors(prior_motion, (0, 0, 1))
    bearings_a_rot = rot.dot(bearings_a)
    bearings_b_rot = rot.dot(bearings_b)
    s, k = 2, 10
    print("S{} K{}".format(s, k))
    bearings_a_norm, _ = solver.normalizer(bearings_a_rot.copy(), s=s, k=k)
    bearings_b_norm, _ = solver.normalizer(bearings_b_rot.copy(), s=s, k=k)
    # bearings_a_norm, bearings_b_norm, _, _ = Solvers.lsq_normalizer(x1=bearings_a_rot, x2=bearings_b_rot)

    A = solver.building_matrix_A(x1=bearings_a, x2=bearings_b)
    A_norm = solver.building_matrix_A(x1=bearings_a_norm, x2=bearings_b_norm)
    _, s, _ = np.linalg.svd(A)
    _, s_n, _ = np.linalg.svd(A_norm)
    print("d: {}, d_norm:{}".format(s[-2], s_n[-2]))

    LB = eval_low_bound_by_observed_matrix(A)
    LB_norm = eval_low_bound_by_observed_matrix(A_norm)
    print("LB: {}, LB:{}".format(LB, LB_norm))

    C = np.linalg.norm(A.T.dot(A), ord="fro")
    C_norm = np.linalg.norm(A_norm.T.dot(A_norm), ord="fro")
    print("C: {}, C_norm:{}".format(C, C_norm))

    d_gap = get_delta_bound_by_bearings(bearings_a, bearings_b)
    d_gap_norm = get_delta_bound_by_bearings(bearings_a_norm, bearings_b_norm)
    print("d-gap: {}, d-gap_norm:{}".format(d_gap, d_gap_norm))

    pm = np.mean(angle_between_vectors_arrays(bearings_a, bearings_b))
    pm_norm = np.mean(angle_between_vectors_arrays(bearings_a_norm, bearings_b_norm))
    print("alpha: {}, alpha_norm:{}".format(pm, pm_norm))
    print("=====================================================================")
