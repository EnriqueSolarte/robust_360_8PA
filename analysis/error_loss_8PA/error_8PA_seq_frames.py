from read_datasets.MP3D_VO import MP3D_VO
from structures.tracker import LKTracker
from structures.extractor.shi_tomasi_extractor import Shi_Tomasi_Extractor
from analysis.utilities.data_utilities import *
from solvers.epipolar_constraint import EightPointAlgorithmGeneralGeometry as g8p
from solvers.epipolar_constraint_by_ransac import RansacEssentialMatrix
from geometry_utilities import evaluate_error_in_transformation
import numpy as np
from plotly.subplots import make_subplots
import os
import plotly.graph_objects as go


def get_file_name(**kwargs):
    scene_ = os.path.dirname(kwargs["data_scene"].scene)
    filename = scene_
    filename += "_res_" + str(kwargs["res"][0]) + "." + str(kwargs["res"][1])
    filename += "_dist" + str(kwargs["distance_threshold"])
    if kwargs["use_ransac"]:
        filename += "_RANSAC_thr" + str(kwargs["residual_threshold"])
        filename += "_trials" + str(kwargs["max_trials"])

    filename += "_" + kwargs["extra"]

    return filename


def plot(**kwargs):
    results = list(kwargs["results"].keys())
    if kwargs["use_ransac"]:
        titles = [dt for dt in results if dt not in ("kf", "features")]
    else:
        titles = [dt for dt in results if dt not in ("kf", "rejections")]
    fig = make_subplots(subplot_titles=titles,
                        rows=2,
                        cols=2,
                        specs=[[{}, {}], [{}, {}]])

    idxs = np.linspace(0, 3, 4).reshape(2, -1)
    for i, dt in enumerate(titles):
        loc = np.squeeze(np.where(idxs == i))
        row, col = loc[0] + 1, loc[1] + 1
        fig.add_trace(go.Scatter(
            x=kwargs["results"]["kf"],
            y=kwargs["results"][dt]),
            row=row, col=col)

        fig.update_xaxes(title_text="Kfrm idx", row=row, col=col)
        fig.update_yaxes(title_text=dt, row=row, col=col)

    fig_file = "{}.html".format(kwargs["filename"])
    fig.update_layout(title_text=fig_file, height=800, width=1800)
    fig.show()
    fig.write_html("plots/{}".format(fig_file))


def eval_function(**kwargs):
    # ! We are saving these variables for every Kf-frm pair
    kwargs["results"] = dict()
    kwargs["results"]["kf"] = []
    kwargs["results"]["error_rot"] = []
    kwargs["results"]["error_tran"] = []
    kwargs["results"]["rejections"] = []
    kwargs["results"]["features"] = []
    kwargs["results"]["residuals"] = []
    ransac = RansacEssentialMatrix(**kwargs)
    solver = g8p()
    while True:
        bearings_kf, bearings_frm, cam_gt, kwargs, ret = track_features(**kwargs)
        if not ret:
            break
        if kwargs.get("use_ransac", False):
            # ! Solving by using RANSAC
            cam_8pa = ransac.solve(data=(
                bearings_kf.copy().T,
                bearings_frm.copy().T)
            )
            num_inliers = sum(ransac.current_inliers)
            num_of_samples = len(ransac.current_inliers)
            kwargs["results"]["rejections"].append(1 - (num_inliers / num_of_samples))
            kwargs["results"]["residuals"].append(ransac.current_residual)
        else:
            # ! SOLVING USING ALL MATCHES
            cam_8pa = solver.recover_pose_from_matches(
                x1=bearings_kf.copy(),
                x2=bearings_frm.copy(),
                eval_current_solution=True
            )
            kwargs["results"]["features"].append(solver.current_count_features)
            kwargs["results"]["residuals"].append(solver.current_residual)

        error = evaluate_error_in_transformation(
            transform_gt=cam_gt,
            transform_est=cam_8pa)
        kwargs["results"]["error_rot"].append(error[0])
        kwargs["results"]["error_tran"].append(error[1])
        kwargs["results"]["kf"].append(kwargs["tracker"].initial_frame.idx)

        print("8PA evaluation - Kf:{} - frm:{}".format(
            kwargs["tracker"].initial_frame.idx,
            kwargs["tracker"].frame_idx
        ))
        print("Error-rot: {}".format(np.median(kwargs["results"]["error_rot"], axis=0)))
        print("Error-tran: {}".format(np.median(kwargs["results"]["error_tran"], axis=0)))

    kwargs["filename"] = "error_8PA_seq_frames_{}.html".format(get_file_name(**kwargs))
    plot(**kwargs)
    save_results(**kwargs)


if __name__ == '__main__':
    path = "/home/kike/Documents/datasets/MP3D_VO"
    scene = "2azQ1b91cZZ/0"
    # scene = "1LXtFkjw3qL/0"
    # scene = "759xd9YjKW5/0"
    # path = "/run/user/1001/gvfs/sftp:host=140.114.27.95,port=50002/NFS/kike/minos/vslab_MP3D_VO/512x1024"
    data = MP3D_VO(scene=scene, basedir=path)

    scene_settings = dict(
        data_scene=data,
        idx_frame=0,
        distance_threshold=0.5,
        res=(360, 180),
        loc=(0, 0),
    )

    ransac_parm = dict(min_samples=8,
                       max_trials=RansacEssentialMatrix.get_number_of_iteration(
                           p_success=0.99, outliers=0.5, min_constraint=8
                       ),
                       residual_threshold=1e-5,
                       verbose=True,
                       use_ransac=True,
                       extra="projected_distance",
                       # extra="sampson_distance",
                       # extra="tangential_distance"
                       )

    features_setting = dict(
        feat_extractor=Shi_Tomasi_Extractor(),
        tracker=LKTracker(),
        show_tracked_features=False
    )

    eval_function(**scene_settings,
                  **features_setting,
                  **ransac_parm)
