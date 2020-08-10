from pcl_utilities import *
from geometry_utilities import *
from Solvers.EpipolarConstraint import EightPointAlgorithmGeneralGeometry_dev as ng8p


def eval_delta_bound_by_fov(v_fov, h_fov, norm=False):
    """
    Returns d-map upper bound by a given FoV
    """
    pcl_main = generate_pcl_by_roi_theta_phi(theta=(-h_fov / 2, h_fov / 2),
                                             phi=(-v_fov / 2, v_fov / 2),
                                             n_pts=10000,
                                             min_d=2,
                                             max_d=20)

    d = []
    solver = ng8p()
    for trial in range(10):
        cam_a2b = get_homogeneous_transform_from_vectors(t_vector=(np.random.uniform(-1, 1),
                                                                   np.random.uniform(-1, 1),
                                                                   np.random.uniform(-1, 1)),
                                                         r_vector=(np.random.uniform(-10, 10),
                                                                   np.random.uniform(-10, 10),
                                                                   np.random.uniform(-10, 10)))

        samples = np.random.randint(0, pcl_main.shape[1], 200)
        bearings_a = sph.sphere_normalization(pcl_main[:, samples])
        bearings_b = sph.sphere_normalization(np.linalg.inv(cam_a2b).dot(pcl_main[:, samples]))
        if norm:
            prior_motion = cam_a2b[0:3, 3]
            rot = get_rot_from_directional_vectors(prior_motion, (0, 0, 1))
            bearings_a_rot = rot.dot(bearings_a)
            bearings_b_rot = rot.dot(bearings_b)

            # bearings_a, _ = Solvers.normalizer(bearings_a_rot.copy(), s=0.2, k=1)
            # bearings_b, _ = Solvers.normalizer(bearings_b_rot.copy(), s=0.2, k=1)
            bearings_a, bearings_b, _, _ = solver.lsq_normalizer(x1=bearings_a, x2=bearings_b)

        A = solver.building_matrix_A(x1=bearings_a, x2=bearings_b)
        u, s, v = np.linalg.svd(A)
        d.append(s[-2])

    return np.nanmean(d)


def eval_bound_by_bearings(x1, x2):
    assert x1.shape == x2.shape
    assert x1.shape[0] == 3

    A = ng8p.building_matrix_A(x1=x1, x2=x2)
    return eval_low_bound_by_observed_matrix(A), eval_upper_bound_by_observed_matrix(A)


def eval_low_bound_by_observed_matrix(matrix):
    assert matrix.shape[1] == 9

    n = matrix.shape[0]
    norm = np.linalg.norm(matrix.T.dot(matrix), ord="fro")
    low_bound = n / (2 * np.sqrt(2))

    return norm - low_bound


def eval_upper_bound_by_observed_matrix(matrix):
    assert matrix.shape[1] == 9

    n = matrix.shape[0]
    norm = np.linalg.norm(matrix.T.dot(matrix), ord="fro")

    return norm ** 2 - n ** 2


def get_frobenius_norm(x1, x2):
    assert x1.shape == x2.shape
    assert x1.shape[0] == 3

    A = ng8p.building_matrix_A(x1=x1, x2=x2)
    c_fro_norm = np.linalg.norm(A.T.dot(A), ord="fro")
    return c_fro_norm


def get_delta_bound_by_bearings(x1, x2):
    assert x1.shape == x2.shape
    assert x1.shape[0] == 3

    n = x1.shape[1]
    c_fro_norm = get_frobenius_norm(x1, x2) ** 2
    # return c_fro_norm
    # print(c_fro_norm)
    sqr_a = (8 * c_fro_norm - n ** 2) / 7

    sqr_b = (n / 8) - (1 / 8) * np.sqrt(sqr_a)
    delta = np.sqrt(sqr_b)
    return delta, c_fro_norm


def get_delta_bound(observed_matrix):
    """
    Returns the evaluation of the delta - gap value, which
    relates sensibility of a 8PA solution given the observed
    matrix A (n,9). n>8 matched points.
    """
    assert observed_matrix.shape[1] == 9

    # ! Equ (21) [Silveira CVPR 19']

    n = observed_matrix.shape[0]
    # ! c_fro_norm has  to be small
    c_fro_norm = np.linalg.norm(observed_matrix.T.dot(observed_matrix), ord="fro") ** 2
    # print(c_fro_norm)
    sqr_a = (8 * c_fro_norm - n ** 2) / 7
    sqr_b = (n / 8) - (1 / 8) * np.sqrt(sqr_a)
    upper_bound_delta = np.sqrt(sqr_b)
    return upper_bound_delta


if __name__ == '__main__':
    fov = (180, 360)
    k_norm = 1
    s_norm = 0.75
    # ! relative camera pose from a to b
    d_, d_norm_ = eval_delta_bound_by_fov(v_fov=fov[0], h_fov=fov[1])
    print(fov)
    print("d     : {0:1.3f}".format(d_))
    print("d_norm: {0:1.3f}".format(d_norm_))
