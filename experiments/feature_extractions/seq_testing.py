import cv2

from config import *

from geometry_utilities import *
from image_utilities import get_mask_map_by_res_loc
from pcl_utilities import *
from read_datasets.MP3D_VO import MP3D_VO
# ! Feature extractor
from structures.extractor.shi_tomasi_extractor import Shi_Tomasi_Extractor
from structures.tracker import LKTracker
from structures.frame import Frame

import plotly.graph_objects as go

error_n8p, error_8p = [], []
matches_list = []

# ! Plot features
fig = go.Figure()


def eval_camera_pose(tracker, cam_gt, output_dir, file):
    # ! relative camera pose from a to b

    from solvers.epipolar_constraint import EightPointAlgorithmGeneralGeometry as g8p
    from solvers.optimal8pa import Optimal8PA as norm_8pa

    g8p_norm = norm_8pa(version=opt_version)
    g8p = g8p()

    cam = Sphere(shape=tracker.initial_frame.shape)
    matches = np.array(tracker.get_matches())
    matches_list.append(matches)

    fig.add_trace(
        go.Scatter(x=matches[0, :, 0],
                   y=matches[0, :, 1],
                   mode='markers',
                   name='Ref'))
    fig.add_trace(
        go.Scatter(x=matches[1, :, 0],
                   y=matches[1, :, 1],
                   mode='markers',
                   name='Key'))
    fig['layout']['xaxis'].update(title='',
                                  range=[0, 1024],
                                  dtick=128,
                                  autorange=False)
    fig.update_xaxes(
        ticktext=list(range(-180, 180 + 1, 45)),
        tickvals=list(range(0, 1024 + 1, 128)),
    )
    fig['layout']['yaxis'].update(title='',
                                  range=[0, 512],
                                  dtick=128,
                                  autorange=False)
    fig.update_yaxes(
        ticktext=list(np.arange(-90, 90 + 1, 22.5)),
        tickvals=list(range(0, 512 + 1, 64)),
    )

    # rot = get_rot_from_directional_vectors((0, 0, 1), (0, 0, 1))
    bearings_kf = cam.pixel2normalized_vector(matches[0])
    bearings_frm = cam.pixel2normalized_vector(matches[1])

    # cam_gt = extend_SO3_to_homogenous(rot).dot(cam_gt).dot(extend_SO3_to_homogenous(rot.T))

    cam_8p = g8p.recover_pose_from_matches(x1=bearings_kf.copy(),
                                           x2=bearings_frm.copy())
    if motion_constraint:
        # ! Forward motion constraint
        prior_motion = cam_8p[0:3, 3]
        rot = get_rot_from_directional_vectors(prior_motion, (0, 0, 1))
        bearings_a_rot = rot.dot(bearings_kf)
        bearings_b_rot = rot.dot(bearings_frm)

        cam_n8p_rot = g8p_norm.recover_pose_from_matches(
            x1=bearings_a_rot.copy(), x2=bearings_b_rot.copy())

        cam_n8p = extend_SO3_to_homogenous(rot.T).dot(cam_n8p_rot).dot(
            extend_SO3_to_homogenous(rot))
    else:
        cam_n8p = g8p_norm.recover_pose_from_matches(x1=bearings_kf.copy(),
                                                     x2=bearings_frm.copy())

        s1 = g8p_norm.T1[0][0]
        k1 = g8p_norm.T1[2][2]
        print("s1, k1 = ({}, {})".format(s1, k1))

        if opt_version != "v1":
            s2 = g8p_norm.T2[0][0]
            k2 = g8p_norm.T2[2][2]
            print("s2, k2 = ({}, {})".format(s2, k2))

    error_n8p.append(
        evaluate_error_in_transformation(transform_gt=cam_gt,
                                         transform_est=cam_n8p))
    error_8p.append(
        evaluate_error_in_transformation(transform_gt=cam_gt,
                                         transform_est=cam_8p))

    print("ours:  {}".format(error_n8p[-1]))
    print("8PA:   {}".format(error_8p[-1]))
    print("kf:{} - frm:{} - matches:{}".format(tracker.initial_frame.idx,
                                               tracker.tracked_frame.idx,
                                               len(tracker.tracks)))
    print(
        "====================================================================="
    )


if __name__ == '__main__':
    assert experiment_group == experiment_group_choices[3] and \
           experiment == experiment_choices[1]

    if dataset == "minos":
        data = MP3D_VO(basedir=basedir, scene=scene)

    tracker = LKTracker()
    threshold_camera_distance = 0.5
    camera_distance = 0
    i = 0

    save = False
    file = None
    mask = None

    # data.number_frames
    errs = []
    for idx in range(i, data.number_frames):
        frame_curr = Frame(**data.get_frame(idx, return_dict=True), idx=idx)

        if idx == i:
            mask = get_mask_map_by_res_loc(data.shape, res=res, loc=(0, 0))

            tracker.set_initial_frame(initial_frame=frame_curr,
                                      extractor=Shi_Tomasi_Extractor(),
                                      mask=mask)
            continue

        relative_pose = frame_curr.get_relative_pose(
            key_frame=tracker.initial_frame)
        camera_distance = np.linalg.norm(relative_pose[0:3, 3])

        if camera_distance > threshold_camera_distance:
            eval_camera_pose(tracker=tracker,
                             cam_gt=relative_pose,
                             output_dir=output_dir,
                             file=file)
            frame_prev = tracker.tracked_frame
            tracker.set_initial_frame(initial_frame=frame_prev,
                                      extractor=Shi_Tomasi_Extractor(),
                                      mask=mask)
            relative_pose = frame_curr.get_relative_pose(
                key_frame=tracker.initial_frame)
            camera_distance = np.linalg.norm(relative_pose[0:3, 3])
            errs.append(idx)

        tracked_img = tracker.track(frame=frame_curr)
        cv2.namedWindow("preview")
        cv2.imshow("preview", tracked_img[:, :, ::-1])
        cv2.waitKey(1)

    fig.data[len(matches_list)].visible = True
    # Create and add slider
    steps = []
    for i in range(len(matches_list)):
        step = dict(
            method="update",
            args=[{
                "visible": [False] * len(matches_list)
            }, {
                "title": str(i)
            }],  # layout attribute
        )
        step["args"][0]["visible"][i] = True  # Toggle i'th trace to "visible"
        steps.append(step)

    sliders = [dict(active=len(matches_list), steps=steps)]

    fig.update_layout(sliders=sliders)

    fig.show()

    print(
        "====================================================================="
    )
    # ! Ours' method
    print("Q1-ours:{} - {}".format(np.quantile(error_n8p, 0.25, axis=0),
                                   len(error_n8p)))
    print("Q2-ours:{} - {}".format(np.median(error_n8p, axis=0),
                                   len(error_n8p)))
    print("Q3-ours:{} - {}".format(np.quantile(error_n8p, 0.75, axis=0),
                                   len(error_n8p)))

    print(
        "====================================================================="
    )
    # ! 8PA
    print("Q1-8PA:{} -  {}".format(np.quantile(error_8p, 0.25, axis=0),
                                   len(error_8p)))
    print("Q2-8PA:{} -  {}".format(np.median(error_8p, axis=0), len(error_8p)))
    print("Q3-8PA:{} -  {}".format(np.quantile(error_8p, 0.75, axis=0),
                                   len(error_8p)))
    print(
        "====================================================================="
    )

    error_8p = np.array(error_8p)
    error_n8p = np.array(error_n8p)

    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    fig = go.Figure()
    fig = make_subplots(rows=1, cols=2)

    # ! Ours Method
    fig.add_trace(go.Scatter(x=errs, y=error_n8p[:, 0], name='ours-rot'),
                  row=1,
                  col=1)
    fig.add_trace(go.Scatter(x=errs, y=error_n8p[:, 1], name='ours-trans'),
                  row=1,
                  col=2)

    # ! 8PA
    fig.add_trace(go.Scatter(x=errs,
                             y=error_8p[:, 0],
                             line=dict(width=2, dash='dot'),
                             name='8p-rot'),
                  row=1,
                  col=1)
    fig.add_trace(go.Scatter(x=errs,
                             y=error_8p[:, 1],
                             line=dict(width=2, dash='dot'),
                             name='8p-trans'),
                  row=1,
                  col=2)

    fig.update_yaxes(title_text="Error", row=1, col=1)
    fig.update_yaxes(title_text="Error", row=1, col=2)
    fig.update_traces(mode='lines+markers', line_shape='linear')
    fig.update_xaxes(title_text=experiment_group[0].upper() +
                     experiment_group[1:] + " - Rot",
                     row=1,
                     col=1)
    fig.update_xaxes(title_text=experiment_group[0].upper() +
                     experiment_group[1:] + " - Trans",
                     row=1,
                     col=2)
    fig.update_layout(title="{}_{}_{}_{}_{}_{}_{}_{}_{}".format(
        experiment, dataset, scene[:-2], scene[-1:],
        "mc" if motion_constraint else "!mc", experiment_group, noise,
        str(res[0]) + "x" + str(res[1]), opt_version),
                      font=dict(
                          family="Courier New, monospace",
                          size=14,
                      ))

    fig.show()

    if save:
        # ! Save .html
        fig.write_html(
            output_dir + "/{}/{}/{}/{}_{}_{}_{}_{}_{}_{}.html".format(
                experiment, dataset, scene, scene[:-2], scene[-1:],
                "mc" if motion_constraint else "!mc", experiment_group, noise,
                str(res[0]) + "x" + str(res[1]), opt_version))

        # ! Save .png
        fig.update_layout(width=1000, height=500)
        fig.write_image(
            output_dir + "/{}/{}/{}/{}_{}_{}_{}_{}_{}_{}.png".format(
                experiment, dataset, scene, scene[:-2], scene[-1:],
                "mc" if motion_constraint else "!mc", experiment_group, noise,
                str(res[0]) + "x" + str(res[1]), opt_version),
            scale=2)
