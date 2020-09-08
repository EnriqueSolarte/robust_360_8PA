"""
The goal of this script is to visualize what are the error_surfaces for a particular pair of
frames (Kf-frm) by using both RANSAC and without RANSAC (i.e., with outliers and with inliers only)
"""
from analysis.utilities import *
from image_utilities import get_mask_map_by_res_loc
from read_datasets.MP3D_VO import MP3D_VO
from solvers.epipolar_constraint_by_ransac import RansacEssentialMatrix
from solvers.epipolar_constraint import EightPointAlgorithmGeneralGeometry
from structures.tracker import LKTracker
from structures.extractor.shi_tomasi_extractor import Shi_Tomasi_Extractor
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from solvers.optimal8pa import Optimal8PA as norm_8pa
from geometry_utilities import *
import os


def get_file_name(**kwargs):
    scene_ = os.path.dirname(kwargs["data_scene"].scene)
    filename = "surface_" + scene_
    filename += "_res_" + str(kwargs["res"][0]) + "." + str(kwargs["res"][1])
    filename += "_dist" + str(kwargs["distance_threshold"])
    filename += "_kf" + str(kwargs["tracker"].initial_frame.idx)
    filename += "_frm" + str(kwargs["tracker"].frame_idx)
    filename += "_grid_" + str(kwargs["grid"][0]) + "." + str(kwargs["grid"][1]) + "." + str(kwargs["grid"][2])
    filename += "_" + kwargs["extra"]
    return filename


def plot_surfaces(**kwargs):
    titles = sorted(list(kwargs["results"].keys()))
    fig = make_subplots(subplot_titles=titles,
                        rows=2,
                        cols=4,
                        specs=[[{'is_3d': True}, {'is_3d': True}, {'is_3d': True}, {'is_3d': True}],
                               [{'is_3d': True}, {'is_3d': True}, {'is_3d': True}, {'is_3d': True}]])

    idxs = np.linspace(0, 7, 8).reshape(2, -1)
    for i, eval in enumerate(titles):
        results = kwargs["results"][eval]
        if eval in kwargs.get("mask_results", (-1,)):
            results = msk(results, kwargs["mask_quantile"])

        loc = np.squeeze(np.where(idxs == i))
        fig.add_trace(go.Surface(x=kwargs["v_grid"],
                                 y=kwargs["v_grid"],
                                 z=results.reshape((len(kwargs["v_grid"]),
                                                    len(kwargs["v_grid"]))),
                                 colorscale='Viridis',
                                 showscale=False),
                      row=loc[0] + 1,
                      col=loc[1] + 1)

    def labels(key):
        return dict(xaxis_title='S',
                    yaxis_title='K',
                    zaxis_title='{}'.format(key))

    fig.update_layout(
        title_text=kwargs["filename"],
        height=800,
        width=1800,
        scene1=labels(titles[0]),
        scene2=labels(titles[1]),
        scene3=labels(titles[2]),
        scene4=labels(titles[3]),
        scene5=labels(titles[4]),
        scene6=labels(titles[5]),
        scene7=labels(titles[6]),
        scene8=labels(titles[7]),
    )
    fig.show()
    fig.write_html("plots/{}.html".format(kwargs["filename"]))
    return kwargs


def eval_function(**kwargs):
    g8p_norm = norm_8pa(version=kwargs.get("opt_version", "v1"))
    g8p = EightPointAlgorithmGeneralGeometry()
    ransac = RansacEssentialMatrix(**kwargs)
    kwargs["mask"] = get_mask_map_by_res_loc(kwargs["data_scene"].shape,
                                             res=kwargs["res"],
                                             loc=kwargs["loc"])
    # ! Getting initial data
    bearings_kf_all, bearings_frm_all, cam_gt, kwargs = track_features(**kwargs)
    e_gt = g8p_norm.build_e_by_cam_pose(cam_gt)

    v = np.linspace(start=kwargs["grid"][0],
                    stop=kwargs["grid"][1],
                    num=kwargs["grid"][2])
    ss, kk = np.meshgrid(v, v)

    kwargs["v_grid"] = v
    kwargs["vv_grid"] = ss.flatten()
    kwargs["results"] = dict()
    # ! No RANSAC results
    kwargs["results"]["all_pts_error_e"] = np.zeros_like(kk.flatten())
    kwargs["results"]["all_pts_error_rot"] = np.zeros_like(kk.flatten())
    kwargs["results"]["all_pts_error_tran"] = np.zeros_like(kk.flatten())
    kwargs["results"]["all_pts_error_residual"] = np.zeros_like(kk.flatten())
    # ! RANSAC results
    kwargs["results"]["inliers_pts_error_e"] = np.zeros_like(kk.flatten())
    kwargs["results"]["inliers_pts_error_rot"] = np.zeros_like(kk.flatten())
    kwargs["results"]["inliers_pts_error_tran"] = np.zeros_like(kk.flatten())
    kwargs["results"]["inliers_pts_error_residual"] = np.zeros_like(kk.flatten())

    cam_no_ransac = g8p.recover_pose_from_matches(
        x1=bearings_kf_all.copy(),
        x2=bearings_frm_all.copy(),
        eval_current_solution=True)
    cam_ransac = ransac.solve(data=(
        bearings_kf_all.copy().T,
        bearings_frm_all.copy().T)
    )
    bearings_kf_inliers = bearings_kf_all[:, ransac.current_inliers]
    bearings_frm_inliers = bearings_frm_all[:, ransac.current_inliers]

    error_ransac = evaluate_error_in_transformation(
        transform_est=cam_ransac,
        transform_gt=cam_gt)
    error_no_ransac = evaluate_error_in_transformation(
        transform_est=cam_no_ransac,
        transform_gt=cam_gt)
    print("Using RANSAC")
    print("error_rot: {}    error_tran: {}".format(error_ransac[0], error_ransac[1]))
    print("rejection ratio: {}".format(ransac.current_rejection_ratio))
    print("Residuals: {}".format(ransac.current_residual))

    print("without RANSAC")
    print("error_rot: {}    error_tran: {}".format(error_no_ransac[0], error_no_ransac[1]))
    print("number of features: {}".format(g8p.current_count_features))
    print("Residuals: {}".format(g8p.current_residual))

    for i in range(len(v) * len(v)):
        S = ss.flatten()[i]
        K = kk.flatten()[i]

        bearings_kf_norm_all, T1 = g8p_norm.normalizer(bearings_kf_all, S, K)
        bearings_frm_norm_all, T2 = g8p_norm.normalizer(bearings_frm_all, S, K)
        e_hat = g8p.compute_essential_matrix(bearings_kf_norm_all,
                                             bearings_frm_norm_all)
        e_hat = np.dot(T2.T, np.dot(e_hat, T1))
        cam_hat = g8p_norm.recover_pose_from_e(e_hat, bearings_kf_all, bearings_frm_all)
        error_cam = evaluate_error_in_transformation(cam_hat, cam_gt)
        kwargs["results"]["all_pts_error_rot"][i] = error_cam[0]
        kwargs["results"]["all_pts_error_tran"][i] = error_cam[1]
        kwargs["results"]["all_pts_error_e"][i] = evaluate_error_in_essential_matrix(e_gt, e_hat)
        kwargs["results"]["all_pts_error_residual"][i] = np.sum(g8p.residual_function_evaluation(
            e=e_hat,
            x1=bearings_kf_norm_all,
            x2=bearings_frm_norm_all))

        bearings_kf_norm_inliers, T1 = g8p_norm.normalizer(bearings_kf_inliers, S, K)
        bearings_frm_norm_inliers, T2 = g8p_norm.normalizer(bearings_frm_inliers, S, K)
        e_hat = g8p.compute_essential_matrix(bearings_kf_norm_inliers,
                                             bearings_frm_norm_inliers)
        e_hat = np.dot(T1.T, np.dot(e_hat, T2))
        cam_hat = g8p_norm.recover_pose_from_e(e_hat, bearings_kf_inliers, bearings_frm_inliers)
        error_cam = evaluate_error_in_transformation(cam_hat, cam_gt)
        kwargs["results"]["inliers_pts_error_rot"][i] = error_cam[0]
        kwargs["results"]["inliers_pts_error_tran"][i] = error_cam[1]
        kwargs["results"]["inliers_pts_error_e"][i] = evaluate_error_in_essential_matrix(e_gt, e_hat)
        kwargs["results"]["inliers_pts_error_residual"][i] = np.sum(g8p.residual_function_evaluation(
            e=e_hat,
            x1=bearings_kf_inliers,
            x2=bearings_frm_inliers))

        kwargs["num_features"] = g8p.current_count_features
        kwargs["ratio_out"] = ransac.current_rejection_ratio
        print("{}: {}".format(get_file_name(**kwargs), i / ss.size))

    kwargs["filename"] = get_file_name(**kwargs)
    save_surface_results(**kwargs)
    plot_surfaces(**kwargs)
    print("done")


if __name__ == '__main__':
    path = "/home/kike/Documents/datasets/MP3D_VO"
    scene = "2azQ1b91cZZ/0"
    # scene = "1LXtFkjw3qL/0"
    # scene = "759xd9YjKW5/0"
    # path = "/run/user/1001/gvfs/sftp:host=140.114.27.95,port=50002/NFS/kike/minos/vslab_MP3D_VO/512x1024"
    data = MP3D_VO(scene=scene, basedir=path)

    scene_settings = dict(
        data_scene=data,
        idx_frame=100,
        distance_threshold=0.5,
        res=(360, 180),
        # res=(180, 180),
        # res=(65.5, 46.4),
        loc=(0, 0),
    )

    model_settings = dict(
        opt_version="v1",
        grid=(-1, 1, 50),
        mask_results=('inliers_pts_error_residual', "inliers_pts_error_rot",
                      "inliers_pts_error_tran", "inliers_pts_error_e",
                      'all_pts_error_e'),
        mask_quantile=0.25,
    )

    ransac_parm = dict(min_samples=8,
                       max_trials=RansacEssentialMatrix.get_number_of_iteration(
                           p_success=0.99, outliers=0.5, min_constraint=8
                       ),
                       residual_threshold=1e-5,
                       verbose=True,
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
                  **ransac_parm,
                  **model_settings)
