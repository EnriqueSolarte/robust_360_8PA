from read_datasets.MP3D_VO import MP3D_VO
from structures.tracker import LKTracker
from structures.extractor.shi_tomasi_extractor import Shi_Tomasi_Extractor
from utilities.data_utilities import *
from solvers.epipolar_constraint_by_ransac import RansacEssentialMatrix
from analysis.sequence_of_frames.camera_recovering import *
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
    filename += "_" + kwargs["extra"]

    return filename


def plot(**kwargs):
    results = list(kwargs["results"].keys())
    titles = [dt for dt in results if dt not in ("kf",)]

    fig = make_subplots(subplot_titles=["Rotation Error", "Trans Error"],
                        rows=2,
                        cols=1,
                        specs=[[{}], [{}]])

    idxs = np.linspace(0, 1, 2).reshape(2, -1)

    dt_results = list()
    dt_results.append([dt for dt in titles if "rot" in dt])
    dt_results.append([dt for dt in titles if "tran" in dt])
    # dt_results.append([dt for dt in titles if "residuals" in dt])

    for i, dt in enumerate(dt_results):
        loc = np.squeeze(np.where(idxs == i))
        row, col = loc[0] + 1, loc[1] + 1
        y_label = dt[0]
        for dt_r in dt:
            if "norm" in dt_r:
                color = COLOR_NORM_8PA
            elif "8pa" in dt_r:
                color = COLOR_8PA
            elif "opt_rpj" in dt_r:
                color = COLOR_OPT_RPJ_RT
            elif "opt_res" in dt_r:
                color = COLOR_OPT_RES_RT

            fig.add_trace(go.Scatter(
                x=kwargs["results"]["kf"],
                y=kwargs["results"][dt_r],
                name=dt_r,
                line=dict(color=color)
            ),
                row=row, col=col)
            if "rot" in dt_r:
                y_label = "Rotation Error"
            elif "tran" in dt_r:
                y_label = "Translation Error"

        fig.update_xaxes(title_text="Kfrm idx", row=row, col=col)
        fig.update_yaxes(title_text=y_label, row=row, col=col)

    fig_file = "{}.html".format(kwargs["filename"])
    fig.update_layout(title_text=fig_file, height=800, width=1800)
    fig.show()
    fig.write_html("plots/{}".format(fig_file))


def run_sequence(**kwargs):
    kwargs["results"] = dict()
    kwargs["results"]["kf"] = []
    kwargs["results"]["norm_8pa_error_rot"] = []
    kwargs["results"]["norm_8pa_error_tran"] = []
    kwargs["results"]["opt_rpj_error_rot"] = []
    kwargs["results"]["opt_rpj_error_tran"] = []

    kwargs["results"]["norm_8pa_reprojection"] = []
    kwargs["results"]["opt_rpj_reprojection"] = []
    kwargs["results"]["opt_res_residuals"] = []

    while True:
        kwargs, ret = get_bearings(**kwargs)
        if not ret:
            break
        kwargs["results"]["kf"].append(kwargs["tracker"].initial_frame.idx)

        # ! 8PA Errors
        cam_hat, residuals = get_cam_pose_by_8pa(**kwargs)
        error = evaluate_error_in_transformation(
            transform_gt=kwargs["cam_gt"],
            transform_est=cam_hat)
        kwargs["results"]["8pa_error_rot"].append(error[0])
        kwargs["results"]["8pa_error_tran"].append(error[1])
        kwargs["results"]["8pa_residuals"].append(np.sum(residuals ** 2))

        # ! Norm 8PA Errors
        cam_hat, residuals = get_cam_pose_by_opt_rpj_norm_8pa(**kwargs)
        error = evaluate_error_in_transformation(
            transform_gt=kwargs["cam_gt"],
            transform_est=cam_hat)
        kwargs["results"]["norm_8pa_error_rot"].append(error[0])
        kwargs["results"]["norm_8pa_error_tran"].append(error[1])
        kwargs["results"]["norm_8pa_residuals"].append(np.sum(residuals ** 2))

        # ! Opt Rt in reprojection 8PA Errors
        cam_hat, residuals = get_cam_pose_by_opt_rpj_rt(**kwargs)
        error = evaluate_error_in_transformation(
            transform_gt=kwargs["cam_gt"],
            transform_est=cam_hat)
        kwargs["results"]["opt_rpj_error_rot"].append(error[0])
        kwargs["results"]["opt_rpj_error_tran"].append(error[1])
        kwargs["results"]["opt_rpj_residuals"].append(np.sum(residuals ** 2))

        # ! Opt Rt residuals 8PA Errors
        cam_hat, residuals = get_cam_pose_by_opt_res_rt(**kwargs)
        error = evaluate_error_in_transformation(
            transform_gt=kwargs["cam_gt"],
            transform_est=cam_hat)
        kwargs["results"]["opt_res_error_rot"].append(error[0])
        kwargs["results"]["opt_res_error_tran"].append(error[1])
        kwargs["results"]["opt_res_residuals"].append(np.sum(residuals ** 2))

        print("----------------------------------------------------------------------------")
        print("8pa Error-rot:           {}".format(np.median(kwargs["results"]["8pa_error_rot"], axis=0)))
        print("norm 8pa Error-rot:      {}".format(np.median(kwargs["results"]["norm_8pa_error_rot"], axis=0)))
        print("Opt Rt rpj Error-rot:    {}".format(np.median(kwargs["results"]["opt_rpj_error_rot"], axis=0)))
        print("Opt Rt res Error-tran:   {}".format(np.median(kwargs["results"]["opt_res_error_rot"], axis=0)))

        print("8pa Error-tran:          {}".format(np.median(kwargs["results"]["8pa_error_tran"], axis=0)))
        print("norm 8pa Error-tran:     {}".format(np.median(kwargs["results"]["norm_8pa_error_tran"], axis=0)))
        print("Opt Rt rpj Error-tran:   {}".format(np.median(kwargs["results"]["opt_rpj_error_tran"], axis=0)))
        print("Opt Rt res Error-tran:   {}".format(np.median(kwargs["results"]["opt_res_error_tran"], axis=0)))

    kwargs["filename"] = "error_8PA_seq_frames_{}".format(get_file_name(**kwargs))
    plot(**kwargs)
    save_results(**kwargs)


if __name__ == '__main__':
    from config import *
    data = MP3D_VO(scene=scene, basedir=basedir)

    scene_settings = dict(
        data_scene=data,
        idx_frame=idx_frame,
        distance_threshold=0.5,
        res=ress[3],
        # res=(180, 180),
        # res=(65.5, 46.4),
        loc=(0, 0),
        extra="test1"
    )

    features_setting = dict(
        feat_extractor=Shi_Tomasi_Extractor(),
        tracker=LKTracker(),
        show_tracked_features=False
    )

    ransac_parm = dict(min_samples=8,
                       max_trials=RansacEssentialMatrix.get_number_of_iteration(
                           p_success=0.99, outliers=0.5, min_constraint=8
                       ),
                       residual_threshold=1e-5,
                       verbose=True,
                       use_ransac=True
                       )

    run_sequence(**scene_settings,
                 **features_setting,
                 **ransac_parm
                 )
