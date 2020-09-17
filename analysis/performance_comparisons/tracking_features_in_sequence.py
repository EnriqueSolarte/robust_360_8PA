from read_datasets.MP3D_VO import MP3D_VO
from structures.tracker import LKTracker
from structures.extractor.shi_tomasi_extractor import Shi_Tomasi_Extractor
from solvers.epipolar_constraint import EightPointAlgorithmGeneralGeometry
from solvers.optimal8pa import Optimal8PA
from analysis.utilities.data_utilities import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def get_file_name(**kwargs):
    scene_ = os.path.dirname(kwargs["data_scene"].scene)
    filename = "Seq_frames_" + scene_
    filename += "_res_" + str(kwargs["res"][0]) + "." + str(kwargs["res"][1])
    filename += "_dist" + str(kwargs["distance_threshold"])
    if kwargs["use_ransac"]:
        filename += "_RANSAC_thr" + str(kwargs["residual_threshold"])
        filename += "_trials" + str(kwargs["max_trials"])
    filename += "_OPT_" + kwargs["opt_version"]
    filename += "_" + kwargs["extra"]

    return filename


def plot(**kwargs):
    results = list(kwargs["results"].keys())
    if kwargs["use_ransac"]:
        titles = [dt for dt in results if dt not in ("kf", "features")]
    else:
        titles = [dt for dt in results if dt not in ("kf", "rejections")]

    fig = make_subplots(subplot_titles=[
        "Rotation Error", "Trans Error", "Residuals", "Features/Rejections"
    ],
                        rows=2,
                        cols=2,
                        specs=[[{}, {}], [{}, {}]])

    idxs = np.linspace(0, 3, 4).reshape(2, -1)

    dt_results = list()
    dt_results.append([dt for dt in titles if "rot" in dt])
    dt_results.append([dt for dt in titles if "tran" in dt])
    dt_results.append([dt for dt in titles if "residuals" in dt])
    dt_results.append(
        [dt for dt in titles if dt == "rejections" or dt == "features"])

    for i, dt in enumerate(dt_results):
        loc = np.squeeze(np.where(idxs == i))
        row, col = loc[0] + 1, loc[1] + 1
        y_label = dt[0]
        for dt_r in dt:
            if "norm" in dt_r:
                color = COLOR_NORM_8PA_OURS
                dash = "solid"
            elif "8pa" in dt_r:
                color = COLOR_8PA
                dash = "dash"
            else:
                color = 'rgb(255,80,127)'
                dash = "solid"

            fig.add_trace(go.Scatter(x=kwargs["results"]["kf"],
                                     y=kwargs["results"][dt_r],
                                     name=dt_r,
                                     line=dict(color=color, dash=dash)),
                          row=row,
                          col=col)
            if "rot" in dt_r:
                y_label = "Rotation Error"
            elif "tran" in dt_r:
                y_label = "Translation Error"
            elif "residuals" in dt_r:
                y_label = "Residuals"

        fig.update_xaxes(title_text="Kfrm idx", row=row, col=col)
        fig.update_yaxes(title_text=y_label, row=row, col=col)

    fig_file = "{}.html".format(kwargs["filename"])
    fig.update_layout(title_text=fig_file, height=800, width=1800)
    fig.show()
    fig.write_html("plots/{}".format(fig_file))


def run_sequence(**kwargs):
    # ! Getting initial data
    ransac = RansacEssentialMatrix(**kwargs)
    g8p = EightPointAlgorithmGeneralGeometry()

    kwargs["results"] = dict()
    kwargs["results"]["kf"] = []
    kwargs["results"]["8pa_error_rot"] = []
    kwargs["results"]["8pa_error_tran"] = []
    kwargs["results"]["8pa_residuals"] = []
    kwargs["results"]["norm_8pa_error_rot"] = []
    kwargs["results"]["norm_8pa_error_tran"] = []
    kwargs["results"]["norm_8pa_residuals"] = []
    kwargs["results"]["rejections"] = []
    kwargs["results"]["features"] = []
    while True:
        bearings_kf, bearings_frm, cam_gt, kwargs, ret = track_features(
            **kwargs)
        if not ret:
            break

        # ! 8PA Evaluation
        if kwargs.get("use_ransac", False):
            # ! Solving by using RANSAC
            cam_8pa = ransac.solve(data=(bearings_kf.copy().T,
                                         bearings_frm.copy().T))
            num_inliers = sum(ransac.current_inliers)
            num_of_samples = len(ransac.current_inliers)
            kwargs["results"]["rejections"].append(1 - (num_inliers /
                                                        num_of_samples))
            kwargs["results"]["8pa_residuals"].append(ransac.current_residual)
            bearings_kf = bearings_kf[:, ransac.current_inliers]
            bearings_frm = bearings_frm[:, ransac.current_inliers]
        else:
            # ! SOLVING USING ALL MATCHES
            cam_8pa = g8p.recover_pose_from_matches(x1=bearings_kf.copy(),
                                                    x2=bearings_frm.copy(),
                                                    eval_current_solution=True)
            kwargs["results"]["features"].append(g8p.current_count_features)
            kwargs["results"]["8pa_residuals"].append(g8p.current_residual)

        norm_8pa = Optimal8PA(kwargs["opt_version"])
        cam_norm_8pa = norm_8pa.recover_pose_from_matches(
            x1=bearings_kf.copy(),
            x2=bearings_frm.copy(),
            eval_current_solution=True)

        kwargs["results"]["norm_8pa_residuals"].append(
            norm_8pa.current_residual)

        kwargs["results"]["kf"].append(kwargs["tracker"].initial_frame.idx)

        # ! 8PA Errors
        error = evaluate_error_in_transformation(transform_gt=cam_gt,
                                                 transform_est=cam_8pa)
        kwargs["results"]["8pa_error_rot"].append(error[0])
        kwargs["results"]["8pa_error_tran"].append(error[1])

        # ! Norm 8PA
        error = evaluate_error_in_transformation(transform_gt=cam_gt,
                                                 transform_est=cam_norm_8pa)
        kwargs["results"]["norm_8pa_error_rot"].append(error[0])
        kwargs["results"]["norm_8pa_error_tran"].append(error[1])

        print("Sequence Info - Kf:{} - frm:{}".format(
            kwargs["tracker"].initial_frame.idx, kwargs["tracker"].frame_idx))
        print("8pa Error-rot: {}".format(
            np.median(kwargs["results"]["8pa_error_rot"], axis=0)))
        print("8pa Error-tran: {}".format(
            np.median(kwargs["results"]["8pa_error_tran"], axis=0)))

        print("norm 8pa Error-rot: {}".format(
            np.median(kwargs["results"]["norm_8pa_error_rot"], axis=0)))
        print("norm 8pa Error-tran: {}".format(
            np.median(kwargs["results"]["norm_8pa_error_tran"], axis=0)))

    kwargs["filename"] = "error_8PA_seq_frames_{}".format(
        get_file_name(**kwargs))
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
        # res=(180, 180),
        # res=(65.5, 46.4),
        loc=(0, 0),
    )

    model_settings = dict(
        opt_version="v1.0.1",
        # extra="epipolar_constraint"
        # extra="projected_distance_C_Sigma_last",
        # extra="sampson_distance",
        # extra="tangential_distance"
        extra="optimal_test")

    ransac_parm = dict(
        min_samples=8,
        max_trials=RansacEssentialMatrix.get_number_of_iteration(
            p_success=0.99, outliers=0.5, min_constraint=8),
        residual_threshold=1e-5,
        verbose=True,
        use_ransac=True)

    features_setting = dict(feat_extractor=Shi_Tomasi_Extractor(),
                            tracker=LKTracker(),
                            show_tracked_features=False)

    run_sequence(**scene_settings, **features_setting, **ransac_parm,
                 **model_settings)
