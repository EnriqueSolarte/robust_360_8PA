from pcl_utilities import *
from solvers.epipolar_constraint import EightPointAlgorithmGeneralGeometry as ng8p
from geometry_utilities import *
from analysis.utilities.optmization_utilities import *


def get_color_by_delta(pcl_size, delta):
    number_pts = pcl_size
    # alpha
    color = np.ones((number_pts, 4)) * 1
    # R color
    color[:, 0] = delta

    # G color
    color[:, 1] = np.ones((number_pts,)) - delta

    #  B color
    color[:, 2] = np.ones((number_pts,)) - delta
    return color


def plot_ovoid_space(v_fov, h_fov, k_norm, s_norm):
    bearings_a, bearings_b = generate_bearings(theta=(-h_fov / 2, h_fov / 2),
                                               phi=(-v_fov / 2, v_fov / 2),
                                               n_pts=20,
                                               # n_pts=50,
                                               min_d=2,
                                               max_d=35,
                                               trans_vector=(0, 0, 0.1),
                                               rot_vector=(0, 0, 0))

    bearings_a_norm, _ = normalizer_Hartley_isotropic(bearings_a)
    bearings_b_norm, _ = normalizer_Hartley_isotropic(bearings_b)

    # ! Here: evaluates the angle between norm-bearings and bearings
    delta_norm = get_angle_between_vectors_arrays(bearings_a_norm, bearings_b_norm)
    delta = get_angle_between_vectors_arrays(bearings_a, bearings_b)

    scale = np.max((delta_norm.max(), delta.max()))
    color = get_color_by_delta(bearings_a.shape[1], delta / scale)
    color_norm = get_color_by_delta(bearings_a.shape[1], delta_norm / scale)

    spatial_shift = np.zeros((3, bearings_a.shape[1]))
    spatial_shift[0, :] = np.ones((bearings_a.shape[1],)) * 5
    pcl = np.hstack((bearings_a, bearings_a_norm + spatial_shift))
    color = np.vstack((color, color_norm))
    plot_color_plc(pcl.T, color, size=0.5, background="white")

    print("done")


if __name__ == '__main__':
    fov = (180, 360)
    k_norm_ = 2.5
    s_norm_ = 2
    plot_ovoid_space(v_fov=fov[0], h_fov=fov[1], k_norm=k_norm_, s_norm=s_norm_)
