from numpy.core.fromnumeric import size
from vispy.util.transforms import translate
from utils.image_utilities import get_color_list
import vispy
from vispy import visuals, scene
import numpy as np
from vispy.visuals.transforms import STTransform, MatrixTransform
from pyquaternion import Quaternion


class CameraPoseVisual:
    def __init__(self, size=0.5, width=2, view=None):
        self.initial_pose = np.eye(4)
        self.width = width
        self.size = size
        # self.axis_z = scene.visuals.create_visual_node(visuals.ArrowVisual)
        self.base = np.zeros((4, 1))
        self.base[3, 0] = 1
        self.view = view
        self.isthereCam = False
        self.prev_pose = np.eye(4)

    def add_camera(self, pose, color):

        self.sphere = scene.visuals.Sphere(
            radius=self.size*0.5,
            # radius=1,
            method='latitude',
            parent=self.view.scene,
            color=color)
        self.initial_pose = pose
        pose_w = pose
        self.sphere.transform = STTransform(translate=pose_w[0:3, 3].T)
        scene.visuals.Arrow
        self.isthereCam = True
        x = self.base + np.array([[self.size], [0], [0], [0]])
        y = self.base + np.array([[0], [self.size], [0], [0]])
        z = self.base + np.array([[0], [0], [self.size], [0]])

        pts = np.hstack([self.base, x, y, z])
        pts = np.dot(pose_w, pts)

        pos = np.zeros((2, 3))
        pos[0, :] = pts[0:3, 0]
        pos[1, :] = pts[0:3, 3]
        self.axis_z = scene.Line(pos=pos, color=(0, 0, 1), method='gl', parent=self.view.scene)
        self.axis_z.set_data(pos=pos, color=(0, 0, 1))

        pos = np.zeros((2, 3))
        pos[0, :] = pts[0:3, 0]
        pos[1, :] = pts[0:3, 1]
        self.axis_x = scene.Line(pos=pos, color=(1, 0, 0), method='gl', parent=self.view.scene)
        self.axis_x.set_data(pos=pos, color=(1, 0, 0))

        pos = np.zeros((2, 3))
        pos[0, :] = pts[0:3, 0]
        pos[1, :] = pts[0:3, 2]
        self.axis_y = scene.Line(pos=pos, color=(0, 1, 0), method='gl', parent=self.view.scene)
        self.axis_y.set_data(pos=pos, color=(0, 1, 0))

        # self.axis_z(pos, width=self.width, color=(1, 1, 1), parent=self.view.scene)

    def transform_camera(self, pose):
        pose_w = pose @ np.linalg.inv(self.initial_pose)
        self.sphere.transform = STTransform(translate=pose[0:3, 3].T)
        q = Quaternion(matrix=pose_w[0:3, 0:3])
        trf = MatrixTransform()
        trf.rotate(angle=np.degrees(q.angle), axis=q.axis)
        trf.translate(pose_w[0:3, 3])
        self.axis_z.transform = trf
        self.axis_x.transform = trf
        self.axis_y.transform = trf


def sphere_frame(view, pose, color=(0.5, 0.2, 1, 1), size=1, width=0.1):

    sphere = scene.visuals.Sphere(radius=size*0.5,
                                  method='latitude',
                                  parent=view.scene,
                                  color=color)
    sphere.transform = STTransform(translate=pose[0:3, 3].T)

    # axis_z = scene.visuals.create_visual_node(visuals.ArrowVisual)

    # base = np.zeros((4, 1))
    # base[3, 0] = 1
    # x = base + np.array([[size], [0], [0], [0]])
    # y = base + np.array([[0], [size], [0], [0]])
    # z = base + np.array([[0], [0], [size], [0]])

    # pts = np.hstack([base, x, y, z])
    # pts = np.dot(pose, pts)

    # pos = np.zeros((2, 3))
    # pos[0, :] = pts[0:3, 0]
    # pos[1, :] = pts[0:3, 1]

    # pos = np.zeros((2, 3))
    # pos[0, :] = pts[0:3, 0]
    # pos[1, :] = pts[0:3, 3]
    # axis_z(pos, width=width, color=(1, 1, 1), parent=view.scene)

    return view


def camera_frame(view, pose, size=1, width=0.1):
    from vispy import visuals, scene
    import numpy as np

    axis_x = scene.visuals.create_visual_node(visuals.LinePlotVisual)
    axis_y = scene.visuals.create_visual_node(visuals.LinePlotVisual)
    axis_z = scene.visuals.create_visual_node(visuals.LinePlotVisual)

    base = np.zeros((4, 1))
    base[3, 0] = 1
    x = base + np.array([[size], [0], [0], [0]])
    y = base + np.array([[0], [size], [0], [0]])
    z = base + np.array([[0], [0], [size], [0]])

    pts = np.hstack([base, x, y, z])
    pts = np.dot(pose, pts)

    pos = np.zeros((2, 3))
    pos[0, :] = pts[0:3, 0]
    pos[1, :] = pts[0:3, 1]
    axis_x(pos, width=width, color='red', parent=view.scene)
    # plot_line(pos, width=width, color='red', parent=view.scene)

    # text = scene.Text("x", parent=view.scene, color='red')
    # text.font_size = 20
    # text.pos = pose[0:3, 3] + np.array((size + 0.5, 0, 0))

    pos = np.zeros((2, 3))
    pos[0, :] = pts[0:3, 0]
    pos[1, :] = pts[0:3, 2]
    axis_y(pos, width=width, color='green', parent=view.scene)
    # plot_line(pos, width=width, color='green', parent=view.scene)

    # text = scene.Text("y", parent=view.scene, color='green')
    # text.font_size = 20
    # text.pos = pose[0:3, 3] + np.array((0, size + 0.5, 0))

    pos = np.zeros((2, 3))
    pos[0, :] = pts[0:3, 0]
    pos[1, :] = pts[0:3, 3]
    axis_z(pos, width=width, color=(0, 0, 0.8), parent=view.scene)
    # plot_line(pos, width=width, color=(0, 0, 0.8), parent=view.scene)

    # text = scene.Text("z", parent=view.scene, color='blue')
    # text.font_size = 20
    # text.pos = pose[0:3, 3] + np.array((0, 0, size + 0.5))

    return view


def camera_sphere(view,
                  pose,
                  size=0.5,
                  alpha=0.5,
                  camera_obj=None,
                  frame=False,
                  return_obj=False,
                  label=None):
    from vispy import scene
    from vispy.visuals.transforms import STTransform

    if camera_obj is not None:
        camera_obj.transform = STTransform(translate=pose[0:3, 3].T)
    else:
        sphere = scene.visuals.Sphere(radius=size,
                                      method='latitude',
                                      parent=view.scene,
                                      color=(1, 1, 1, alpha))
        sphere.transform = STTransform(translate=pose[0:3, 3].T)
    if frame:
        camera_frame(view=view, pose=pose, size=size * 2)

    if label is not None:
        text = scene.Text(label, parent=view.scene, color='white')
        text.font_size = 10
        text.pos = pose[0:3, 3].T

    if return_obj:
        return view, sphere
    return view


def sphere(view, pose, size=1, alpha=0.8, color=None):
    from vispy import scene
    from vispy.visuals.transforms import STTransform
    if color is None:
        color = (0.5, 0.5, 0.1, alpha)

    try:
        loc = pose[0:3, 3].T
    except:
        loc = pose
    sphere = scene.visuals.Sphere(radius=size,
                                  method='latitude',
                                  parent=view.scene,
                                  color=color)
    sphere.transform = STTransform(translate=loc)

    return view


def camera_plane(view,
                 pose,
                 shape,
                 focal,
                 size=0.1,
                 color='blue',
                 frame=False):
    import numpy as np
    f = 1 * size
    h, w = size * shape[0] / focal, size * shape[0] / focal

    base = np.zeros((4, 1))
    base[3, 0] = 1
    p1 = base + np.array([[w / 2], [-h / 2], [f], [0]])
    p2 = base + np.array([[w / 2], [h / 2], [f], [0]])
    p3 = base + np.array([[-w / 2], [h / 2], [f], [0]])
    p4 = base + np.array([[-w / 2], [-h / 2], [f], [0]])

    pts = np.hstack([base, p1, p2, p3, p4])
    pts = np.dot(pose, pts)

    pos = np.zeros((2, 3))
    pos[0, :] = pts[0:3, 0]
    pos[1, :] = pts[0:3, 1]
    draw_line(view=view, points=pos, color=color)

    pos = np.zeros((2, 3))
    pos[0, :] = pts[0:3, 0]
    pos[1, :] = pts[0:3, 2]
    draw_line(view=view, points=pos, color=color)

    pos = np.zeros((2, 3))
    pos[0, :] = pts[0:3, 0]
    pos[1, :] = pts[0:3, 3]
    draw_line(view=view, points=pos, color=color)

    pos = np.zeros((2, 3))
    pos[0, :] = pts[0:3, 0]
    pos[1, :] = pts[0:3, 4]
    draw_line(view=view, points=pos, color=color)

    pos = np.zeros((2, 3))
    pos[0, :] = pts[0:3, 1]
    pos[1, :] = pts[0:3, 2]
    draw_line(view=view, points=pos, color=color)
    pos = np.zeros((2, 3))
    pos[0, :] = pts[0:3, 2]
    pos[1, :] = pts[0:3, 3]
    draw_line(view=view, points=pos, color=color)
    pos = np.zeros((2, 3))
    pos[0, :] = pts[0:3, 3]
    pos[1, :] = pts[0:3, 4]
    draw_line(view=view, points=pos, color=color)

    pos = np.zeros((2, 3))
    pos[0, :] = pts[0:3, 4]
    pos[1, :] = pts[0:3, 1]
    draw_line(view=view, points=pos, color=color)

    if frame:
        camera_frame(view=view, pose=pose, size=size / 2)

    return view


def setting_viewer(return_canvas=False, main_axis=True, bgcolor='black'):
    import vispy.scene
    from vispy.scene import visuals

    canvas = vispy.scene.SceneCanvas(keys='interactive',
                                     show=True,
                                     bgcolor=bgcolor)
    size_win = 1024
    canvas.size = 2*size_win, size_win

    view = canvas.central_widget.add_view()
    view.camera = 'arcball'  # turntable / arcball / fly / perspective

    if main_axis:
        visuals.XYZAxis(parent=view.scene)

    if return_canvas:
        return view, canvas
    return view


def setting_pcl(view, size=5, edge_width=2, antialias=0):
    from vispy.scene import visuals
    from functools import partial
    scatter = visuals.Markers()
    scatter.set_gl_state('translucent',
                         depth_test=True,
                         blend=True,
                         blend_func=('src_alpha', 'one_minus_src_alpha'))
    # scatter.set_gl_state(depth_test=True)
    scatter.antialias = 0
    view.add(scatter)
    return partial(scatter.set_data, size=size, edge_width=edge_width)


def draw_line(view, points, color):
    # from vispy import visuals, scene

    line = scene.visuals.create_visual_node(visuals.LinePlotVisual)
    line(points, width=2.0, color=color, parent=view.scene)
    return view


def plot_color_plc(points,
                   color=(1, 1, 1, 1),
                   return_view=False,
                   size=0.5,
                   plot_main_axis=True,
                   background="black",
                   axis_frame=None,
                   scale_factor=15
                   ):
    import vispy
    from functools import partial

    # if color == (1, 1, 1):
    #     bg = "black"
    view = setting_viewer(main_axis=plot_main_axis, bgcolor=background)

    # view.camera = vispy.scene.TurntableCamera(elevation=45,
    #                                           azimuth=-135,
    #                                           roll=0,
    #                                           fov=0,
    #                                           up='-y')
    view.camera = vispy.scene.TurntableCamera(elevation=90,
                                              azimuth=0,
                                              roll=0,
                                              fov=0,
                                              up='-y')
    # view.camera = vispy.scene.TurntableCamera(elevation=0, azimuth=180, roll=0, fov=0, up='-y')
    view.camera.scale_factor = scale_factor
    draw_pcl = setting_pcl(view=view)
    draw_pcl(points, edge_color=color, size=size)
    # pose1 = np.eye(4)
    # sphere(view, pose1, size=1, alpha=0.5)
    # pose1[0, 3] = axis_frame[0]
    # camera_frame(view, pose1, size=axis_frame, width=2)
    # pose1[0, 3] = axis_frame[1]
    # camera_frame(view, pose1, size=3, width=2)
    if not return_view:
        vispy.app.run()
    else:
        return view


def plot_multiples_lines(list_lines, color=(1, 1, 1, 1),
                         size=1,
                         plot_main_axis=True,
                         background="black",
                         scale_factor=10):
    from vispy import visuals, scene, app
    # build visuals
    plot3D = scene.visuals.create_visual_node(visuals.LinePlotVisual)

    # build canvas
    canvas = scene.SceneCanvas(keys='interactive', title='Layout', show=True)

    # Add a ViewBox to let the user zoom/rotate
    view = canvas.central_widget.add_view()
    view.camera = scene.TurntableCamera(elevation=90,
                                        azimuth=100,
                                        roll=0,
                                        fov=0,
                                        up='-y')
    view.camera.scale_factor = scale_factor
    scene.visuals.XYZAxis(parent=view.scene)

    for points in list_lines:
        if len(points.shape) < 2:
            assert points.shape[0] == 3
            points = np.vstack(((0, 0, 0), points))
        else:
            assert points.shape == (2, 3)

        plot3D(points, width=size, color=color,
               edge_color='w', symbol='o', face_color=(0.2, 0.2, 1, 0.8),
               parent=view.scene)
    app.run()


def plot_line(points,
              color=(1, 1, 1, 1),
              size=1,
              plot_main_axis=True,
              background="black",
              scale_factor=10
              ):
    from vispy import visuals, scene, app

    if len(points.shape) < 2:
        assert points.shape[0] == 3
        points = np.vstack(((0, 0, 0), points))
    else:
        assert points.shape == (2, 3)

    # build visuals
    plot3D = scene.visuals.create_visual_node(visuals.LinePlotVisual)

    # build canvas
    canvas = scene.SceneCanvas(keys='interactive', title='Layout', show=True)

    # Add a ViewBox to let the user zoom/rotate
    view = canvas.central_widget.add_view()
    view.camera = scene.TurntableCamera(elevation=90,
                                        azimuth=100,
                                        roll=0,
                                        fov=0,
                                        up='-y')
    view.camera.scale_factor = scale_factor
    scene.visuals.XYZAxis(parent=view.scene)

    plot3D(points, width=size, color=color,
           edge_color='w', symbol='o', face_color=(0.2, 0.2, 1, 0.8),
           parent=view.scene)
    app.run()


def plot_pcl_and_cameras(points,
                         color=(1, 1, 1),
                         return_view=False,
                         size=1,
                         plot_main_axis=True,
                         background="white",
                         cam1=np.eye(4),
                         cam2=np.eye(4)):
    import vispy
    view = setting_viewer(main_axis=plot_main_axis, bgcolor=background)

    view.camera = vispy.scene.TurntableCamera(elevation=90,
                                              azimuth=100,
                                              roll=0,
                                              fov=0,
                                              up='-y')
    # view.camera = vispy.scene.TurntableCamera(elevation=0, azimuth=180, roll=0, fov=0, up='-y')
    view.camera.scale_factor = 10

    draw_pcl = setting_pcl(view=view)
    draw_pcl(points, edge_color=color, size=size)
    camera_sphere(view, cam1, size=0.5, alpha=0.1, frame=True)
    camera_sphere(view, cam2, size=0.5, alpha=0.1, frame=True)

    # pose1[0, 3] = 4
    # camera_frame(view, pose1, size=1, width=2)
    if not return_view:
        vispy.app.run()
    else:
        return view


def plot_pcl_list(list_pcl):
    dim, _ = list_pcl[0].shape
    len_ = len(list_pcl)
    plc_ = np.hstack(list_pcl)[0:3, :]
    colors = get_color_list(list_pcl)
    colors_ = []
    for c, pts in zip(colors.T, list_pcl):
        cc = np.ones_like(pts)*c.reshape(-1, 1)
        colors_.append(cc)

    colors_ = np.hstack(colors_)
    plot_color_plc(plc_.reshape((3, -1)).T, colors_.reshape((3, -1)).T)
