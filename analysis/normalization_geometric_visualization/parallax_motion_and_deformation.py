from pcl_utilities import *
from solvers.epipolar_constraint import EightPointAlgorithmGeneralGeometry as ng8p
from geometry_utilities import *
from analysis.utilities.optmization_utilities import *
from vispy import app, scene, io


def plot_spheres(bearings, color, norm_bearings, norm_color):
    canvas = vispy.scene.SceneCanvas(keys='interactive')
    canvas.size = 1024, 1024
    canvas.show()

    # Create two ViewBoxes, place side-by-side
    vb1 = scene.widgets.ViewBox(parent=canvas.scene)
    vb2 = scene.widgets.ViewBox(parent=canvas.scene)
    #
    grid = canvas.central_widget.add_grid()
    grid.padding = 6
    grid.add_widget(vb1, 0, 0)
    grid.add_widget(vb2, 0, 1)

    vb1.camera = vispy.scene.TurntableCamera(elevation=45,
                                             azimuth=-135,
                                             roll=0,
                                             fov=0,
                                             up='-y')
    vb2.camera = vispy.scene.TurntableCamera(elevation=45,
                                             azimuth=-135,
                                             roll=0,
                                             fov=0,
                                             up='-y')

    vb1.camera.scale_factor = 7
    vb2.camera.scale_factor = 7

    draw_pcl = setting_plc(view=vb1)
    draw_pcl(bearings, edge_color=color, size=2)
    camera_frame(vb1, np.eye(4), size=2, width=5)
    draw_pcl = setting_plc(view=vb2)
    draw_pcl(norm_bearings, edge_color=norm_color, size=2)
    camera_frame(vb2, np.eye(4), size=3, width=5)

    vb1.camera.link(vb2.camera)
    vispy.app.run()


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
                                               n_pts=1000000,
                                               # n_pts=50,
                                               min_d=33,
                                               max_d=35,
                                               trans_vector=(0.01, 0.01, 0.01),
                                               rot_vector=(0, 0, 0))

    bearings_a_norm, _ = normalizer_s_k(bearings_a, s=s_norm, k=k_norm)
    bearings_b_norm, _ = normalizer_s_k(bearings_b, s=s_norm, k=k_norm)

    # ! Here: evaluates the angle between norm-bearings and bearings
    delta_norm = get_angle_between_vectors_arrays(bearings_a_norm, bearings_b_norm)
    delta = get_angle_between_vectors_arrays(bearings_a, bearings_b)

    scale = np.max((delta_norm.max(), delta.max()))
    color = get_color_by_delta(bearings_a.shape[1], delta / scale)
    color_norm = get_color_by_delta(bearings_a.shape[1], delta_norm / scale)

    spatial_shift = np.zeros((3, bearings_a.shape[1]))
    distance = 10
    spatial_shift[0, :] = np.ones((bearings_a.shape[1],)) * distance
    # spatial_shift[0, :] = np.ones((bearings_a.shape[1],)) * distance
    pcl = np.hstack((bearings_a, bearings_a_norm + spatial_shift))
    # color = np.vstack((color, color_norm))
    # plot_color_plc(pcl.T, color, size=0.5, background="black", axis_frame=(0, distance))
    plot_spheres(bearings_a.T, color, bearings_a_norm.T, color_norm)
    # plot_color_plc(bearings_a.T, color, size=0.5, background="black", axis_frame=2)
    # plot_color_plc(bearings_a_norm.T, color_norm, size=0.5, background="black", axis_frame=3)

    print("done")


if __name__ == '__main__':
    fov = (180, 360)
    k_norm_ = 1
    s_norm_ = 2
    plot_ovoid_space(v_fov=fov[0], h_fov=fov[1], k_norm=k_norm_, s_norm=s_norm_)
