from solvers.optimal8pa import Optimal8PA as norm_8pa
from solvers.epipolar_constraint import EightPointAlgorithmGeneralGeometry as g8pa
from pcl_utilities import *
from read_datasets.MP3D_VO import MP3D_VO
from geometry_utilities import *
from image_utilities import get_mask_map_by_res_loc
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
import os
from structures.extractor.orb_extractor import ORBExtractor
from structures.extractor.shi_tomasi_extractor import Shi_Tomasi_Extractor
import cv2
from analysis.delta_bound import get_frobenius_norm

pio.renderers.default = "browser"

OURS_COLOR = (200, 100, 51)
_8PA_COLOR = (0, 0, 154)
MIN_COLOR = (0, 255, 50)


def pcl_creation(**kwargs):
    data_scene = kwargs["data_scene"]
    assert kwargs["pcl"] in ("by_sampling", "by_k_features")
    if kwargs["pcl"] == "by_sampling":
        pcl_dense, pcl_dense_color, _ = data_scene.get_pcl(
            idx=kwargs["idx_frame"])
        pcl, mask = mask_pcl_by_res_and_loc(pcl=pcl_dense,
                                            loc=kwargs["loc"],
                                            res=kwargs["res"])
    else:
        mask = get_mask_map_by_res_loc(data_scene.shape,
                                       res=kwargs["res"],
                                       loc=kwargs["loc"])
        pcl = data_scene.get_pcl_from_key_features(
            idx=kwargs["idx_frame"],
            extractor=kwargs["feat_extractor"],
            mask=mask)

    if pcl.shape[1] > kwargs["max_pts"]:
        samples = np.random.randint(0, pcl.shape[1], kwargs["max_pts"])
        pcl = pcl[:, samples]

    return pcl


def get_file_name(**kwargs):
    scene = os.path.dirname(kwargs["data_scene"].scene)
    filename = scene + "_plc_" + kwargs["pcl"] + "_fr_" + str(
        kwargs["idx_frame"])
    filename += "_fov_" + str(kwargs["res"][0]) + "." + str(kwargs["res"][1])
    filename += "_noise_" + str(kwargs["noise"]) + "." + str(
        kwargs["outliers"])
    filename += "_grid_" + str(kwargs["grid"][0])
    filename += "." + str(kwargs["grid"][1])
    filename += "." + str(kwargs["grid"][2])

    if None in kwargs["mask_results"]:
        pass
    else:
        filename += "_masking_"
        for mask in kwargs["mask_results"]:
            filename += mask + "_"
        filename += str(kwargs["mask_quantile"])

    return filename


def msk(eval, quantile):
    pivot = np.quantile(eval, quantile)
    # pivot = np.inf
    mask = eval > pivot
    eval[mask] = pivot
    return eval


def eval_error_surface(**kwargs):
    g8p_norm = norm_8pa(version=kwargs["opt_version"])
    pcl = pcl_creation(**kwargs)

    bearings_a, bearings_b, cam_a2b = get_bearings_from_pcl(
        pcl=pcl,
        t_vector=kwargs["t_vector"],
        rotation=kwargs["r_vector"],
        noise=kwargs["noise"],
        outliers=kwargs["outliers"] * pcl.shape[1])
    plot_pcl_and_cameras(pcl[0:3, :].T, cam2=cam_a2b)

    e = g8p_norm.build_e_by_cam_pose(cam_a2b)
    v = np.linspace(start=kwargs["grid"][0],
                    stop=kwargs["grid"][1],
                    num=kwargs["grid"][2])
    ss, kk = np.meshgrid(v, v)

    kwargs["v_grid"] = v
    kwargs["vv_grid"] = ss.flatten()
    kwargs["losses"] = dict()
    kwargs["losses"]["error_e"] = np.zeros_like(kk.flatten())
    kwargs["losses"]["error_rot"] = np.zeros_like(kk.flatten())
    kwargs["losses"]["error_tran"] = np.zeros_like(kk.flatten())
    kwargs["losses"]["loss_c"] = np.zeros_like(kk.flatten())
    kwargs["losses"]["loss_delta"] = np.zeros_like(kk.flatten())
    kwargs["losses"]["loss_pm"] = np.zeros_like(kk.flatten())

    cam_hat_8pa = g8pa().recover_pose_from_matches(x1=bearings_a.copy(),
                                                   x2=bearings_b.copy())
    error_cam_8pa = evaluate_error_in_transformation(cam_hat_8pa, cam_a2b)

    for i in range(len(v) * len(v)):
        S = ss.flatten()[i]
        K = kk.flatten()[i]

        bearings_a_norm, T1 = g8p_norm.normalizer(bearings_a, S, K)
        bearings_b_norm, T2 = g8p_norm.normalizer(bearings_b, S, K)
        e_hat = g8p_norm.compute_essential_matrix(bearings_a_norm,
                                                  bearings_b_norm)
        e_hat = np.dot(T1.T, np.dot(e_hat, T2))
        cam_a2b_hat = g8p_norm.recover_pose_from_e(e_hat, bearings_a, bearings_b)
        error_cam = evaluate_error_in_transformation(cam_a2b_hat, cam_a2b)
        kwargs["losses"]["error_rot"][i] = error_cam[0]
        kwargs["losses"]["error_tran"][i] = error_cam[1]
        # ! Using Spherical bearing only for testing.
        # ! By using unit-bearings we can evaluate the internal angles as Silveira [CVPR'19]
        x1 = spherical_normalization(bearings_a_norm)
        x2 = spherical_normalization(bearings_b_norm)
        C_2 = get_frobenius_norm(x1, x2)

        C, A = get_frobenius_norm(bearings_a_norm,
                                  bearings_b_norm,
                                  return_A=True)
        kwargs["losses"]["loss_c"][i] = C
        kwargs["losses"]["loss_pm"][i] = np.nanmean(
            get_angle_between_vectors_arrays(bearings_a_norm, bearings_b_norm))
        u, sigma, v = np.linalg.svd(A)
        kwargs["losses"]["loss_delta"][i] = sigma[-2]
        kwargs["losses"]["error_e"][i] = evaluate_error_in_essential_matrix(
            e, e_hat)
        print("{}:{}".format(get_file_name(**kwargs), i / ss.size))

    kwargs["losses"]["error_cam"] = eval_cam_error(
        kwargs["losses"]["error_rot"], kwargs["losses"]["error_tran"])
    kwargs["losses"]["loss"] = g8p_norm.loss(
        C=kwargs["losses"]["loss_c"],
        delta=kwargs["losses"]["loss_delta"],
        pm=kwargs["losses"]["loss_pm"])

    cam_hat_n8pa = g8p_norm.recover_pose_from_matches(
        x1=bearings_a.copy(),
        x2=bearings_b.copy(),
        param=kwargs["optimal_parameters"])
    error_cam_n8pa = evaluate_error_in_transformation(cam_hat_n8pa, cam_a2b)

    print("Our camera error:Rot={}    Trans={}".format(error_cam_n8pa[0],
                                                       error_cam_n8pa[1]))
    print("8PA camera error:Rot={}    Trans={}".format(error_cam_8pa[0],
                                                       error_cam_8pa[1]))
    kwargs["Ours"] = dict()
    kwargs["Ours"]["error_rot"] = error_cam_n8pa[0]
    kwargs["Ours"]["error_tran"] = error_cam_n8pa[1]
    kwargs["Ours"]["S"] = g8p_norm.T1[0, 0]
    kwargs["Ours"]["K"] = g8p_norm.T1[2, 2]

    kwargs["8PA"] = dict()
    kwargs["8PA"]["error_rot"] = error_cam_8pa[0]
    kwargs["8PA"]["error_tran"] = error_cam_8pa[1]

    kwargs = plot_contours(**kwargs)
    plot_surfaces(**kwargs)


def eval_cam_error(rot_error, tra_error):
    return 0.5 * (rot_error / np.linalg.norm(rot_error) +
                  tra_error / np.linalg.norm(tra_error))


def plot_contours(**kwargs):
    titles = sorted(list(kwargs["losses"].keys()))
    fig = make_subplots(subplot_titles=titles,
                        rows=2,
                        cols=4,
                        specs=[[{}, {}, {}, {}], [{}, {}, {}, {}]])

    idxs = np.linspace(0, 7, 8).reshape(2, -1)
    kwargs["minimum"] = dict()
    for i, eval in enumerate(titles):
        # ! get min values
        results = kwargs["losses"][eval].reshape(
            (len(kwargs["v_grid"]), len(kwargs["v_grid"])))
        min_val = np.unravel_index(np.argmin(results, axis=None),
                                   results.shape)
        kwargs["minimum"][eval] = kwargs["v_grid"][
            min_val[1]], kwargs["v_grid"][min_val[0]], results.min()
        if eval in kwargs["mask_results"]:
            results = msk(results, kwargs["mask_quantile"])
        loc = np.squeeze(np.where(idxs == i))
        fig.add_trace(go.Contour(x=kwargs["v_grid"],
                                 y=kwargs["v_grid"],
                                 z=results,
                                 colorscale='Viridis',
                                 showscale=False),
                      row=loc[0] + 1,
                      col=loc[1] + 1)

        fig.update_xaxes(title_text="S", row=loc[0] + 1, col=loc[1] + 1)
        fig.update_yaxes(title_text="K", row=loc[0] + 1, col=loc[1] + 1)
        fig.add_trace(go.Scatter(x=(1, ),
                                 y=(1, ),
                                 mode='markers',
                                 marker=dict(size=8, color=_8PA_COLOR),
                                 name="8PA"),
                      row=loc[0] + 1,
                      col=loc[1] + 1)
        fig.add_trace(go.Scatter(x=(kwargs["minimum"][eval][0], ),
                                 y=(kwargs["minimum"][eval][1], ),
                                 name="min",
                                 mode='markers',
                                 marker=dict(size=10, color=MIN_COLOR)),
                      row=loc[0] + 1,
                      col=loc[1] + 1)
        fig.add_trace(go.Scatter(x=(kwargs["Ours"]["S"], ),
                                 y=(kwargs["Ours"]["K"], ),
                                 name="Ours",
                                 mode='markers',
                                 marker=dict(size=8, color=OURS_COLOR)),
                      row=loc[0] + 1,
                      col=loc[1] + 1)

    fig_file = "contour_{}.html".format(get_file_name(**kwargs))

    fig.update_layout(title_text=fig_file, height=800, width=1800)
    fig.show()
    fig.write_html("plots/{}".format(fig_file))
    return kwargs


def plot_surfaces(**kwargs):
    titles = sorted(list(kwargs["losses"].keys()))
    fig = make_subplots(subplot_titles=titles,
                        rows=2,
                        cols=4,
                        specs=[[{
                            'is_3d': True
                        }, {
                            'is_3d': True
                        }, {
                            'is_3d': True
                        }, {
                            'is_3d': True
                        }],
                               [{
                                   'is_3d': True
                               }, {
                                   'is_3d': True
                               }, {
                                   'is_3d': True
                               }, {
                                   'is_3d': True
                               }]])

    idxs = np.linspace(0, 7, 8).reshape(2, -1)
    for i, eval in enumerate(titles):
        results = kwargs["losses"][eval]
        if eval in kwargs["mask_results"]:
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

        if eval in ("error_rot", "error_tran"):
            fig.add_trace(go.Scatter3d(x=(kwargs["Ours"]["S"], ),
                                       y=(kwargs["Ours"]["K"], ),
                                       z=(kwargs["Ours"][eval], ),
                                       marker=dict(color=OURS_COLOR, size=5),
                                       name="Ours"),
                          row=loc[0] + 1,
                          col=loc[1] + 1)
            fig.add_trace(go.Scatter3d(x=(1, ),
                                       y=(1, ),
                                       z=(kwargs["8PA"][eval], ),
                                       marker=dict(color=_8PA_COLOR, size=5),
                                       name="8PA"),
                          row=loc[0] + 1,
                          col=loc[1] + 1)
            fig.add_trace(go.Scatter3d(x=(kwargs["minimum"][eval][0], ),
                                       y=(kwargs["minimum"][eval][1], ),
                                       z=(kwargs["minimum"][eval][2], ),
                                       marker=dict(color=MIN_COLOR, size=5),
                                       name="min"),
                          row=loc[0] + 1,
                          col=loc[1] + 1)

    def labels(key):
        return dict(xaxis_title='S',
                    yaxis_title='K',
                    zaxis_title='{}'.format(key))

    fig_file = "surface_{}.html".format(get_file_name(**kwargs))

    fig.update_layout(
        title_text=fig_file,
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
    # fig.update_traces(contours_z=dict(show=True, usecolormap=True,
    #                                   highlightcolor="limegreen", project_z=True))
    fig.show()
    fig.write_html("plots/{}".format(fig_file))
    return kwargs


if __name__ == '__main__':
    np.random.seed(50)
    path = "/home/kike/Documents/datasets/MP3D_VO"
    scene = "2azQ1b91cZZ/0"
    # scene = "1LXtFkjw3qL/0"
    # scene = "759xd9YjKW5/0"
    # path = "/run/user/1001/gvfs/sftp:host=140.114.27.95,port=50002/NFS/kike/minos/vslab_MP3D_VO/512x1024"
    data = MP3D_VO(scene=scene, basedir=path)

    scene_settings = dict(
        data_scene=data,
        idx_frame=549,
        max_pts=150,
        res=(65.5, 46.4),
        # res=(180, 180),
        loc=(0, 0),
        # feat_extractor=ORBExtractor(),
        feat_extractor=Shi_Tomasi_Extractor())

    pcl_settings = dict(
        pcl="by_k_features",
        # pcl="by_sampling",
        t_vector=(0.01, 0, 0.5),
        r_vector=(10, 10, 10),
        noise=500,
        outliers=0.05)

    model_settings = dict(
        opt_version="v1",
        grid=(-1, 1, 100),
        mask_results=("loss", "loss_delta", "loss_c", "error_rot",
                      "error_tran"),
        # mask_results=(None,),
        mask_quantile=0.25,
        optimal_parameters=None)

    eval_error_surface(**scene_settings, **pcl_settings, **model_settings)
