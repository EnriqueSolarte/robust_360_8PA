from solvers.optimal8pa import Optimal8PA as norm_8pa
from pcl_utilities import *
from read_datasets.MP3D_VO import MP3D_VO
from geometry_utilities import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
import os

pio.renderers.default = "browser"

from analysis.delta_bound import get_frobenius_norm


def msk(eval, quantile):
    pivot = np.quantile(eval, quantile)
    # pivot = np.inf
    mask = eval > pivot
    eval[mask] = pivot
    return eval


def eval_loss(c, delta, pm, quantile):
    # cnst = (0.01, 1, 1)
    # return msk(cnst[0] * c + cnst[1] * (1 / delta) + cnst[2] * (1 / pm))
    return msk((0.5 * c + 1 / pm + 1 / delta), quantile=quantile)


def contour_plot(s, k, e_error, rot_e, tran_e, loss_c, loss_delta, loss_pm, **kwargs):
    loss = eval_loss(c=loss_c, delta=loss_delta, pm=loss_pm, quantile=kwargs["figure_specs"]["QX"])
    print("done")

    fig = make_subplots(
        subplot_titles=("E-error", "Rot-e", "Tran-e", "T-e",
                        "loss C", "loss delta", "loss Omega", "loss"),
        rows=2, cols=4,
        specs=[[{}, {}, {}, {}],
               [{}, {}, {}, {}]])

    fig.add_trace(
        go.Contour(x=s, y=k, z=e_error.reshape((len(s), len(s))),
                   colorscale='Viridis',
                   showscale=False),
        row=1, col=1)

    fig.add_trace(
        go.Contour(x=s, y=k, z=rot_e.reshape(len(s), len(s)),
                   colorscale='Viridis',
                   showscale=False),
        row=1, col=2)

    fig.add_trace(
        go.Contour(x=s, y=k, z=tran_e.reshape(len(s), len(s)),
                   colorscale='Viridis',
                   showscale=False),
        row=1, col=3)

    t_e = rot_e / np.linalg.norm(rot_e) + tran_e / np.linalg.norm(tran_e)
    fig.add_trace(
        go.Contour(x=s, y=k, z=t_e.reshape(len(s), len(s)),
                   colorscale='Viridis',
                   showscale=False),
        row=1, col=4)

    fig.add_trace(
        go.Contour(x=s, y=k, z=loss_c.reshape(len(s), len(s)),
                   colorscale='Viridis',
                   showscale=False),
        row=2, col=1)

    fig.add_trace(
        go.Contour(x=s, y=k, z=loss_delta.reshape(len(s), len(s)),
                   colorscale='Viridis',
                   showscale=False),
        row=2, col=2)

    fig.add_trace(
        go.Contour(x=s, y=k, z=loss_pm.reshape(len(s), len(s)),
                   colorscale='Viridis',
                   showscale=False),
        row=2, col=3)

    fig.add_trace(
        go.Contour(x=s, y=k, z=loss.reshape(len(s), len(s)),
                   colorscale='Viridis',
                   showscale=False),
        row=2, col=4)

    # Update xaxis properties
    for row in range(2):
        for col in range(4):
            fig.update_xaxes(title_text="S", row=row + 1, col=col + 1)
            fig.update_yaxes(title_text="K", row=row + 1, col=col + 1)
            fig.add_trace(
                go.Scatter(x=(1,), y=(1,), mode='markers', marker=dict(size=5)),
                row=row + 1, col=col + 1)

    fig_file = "contour_{}.{}_{}_{}_{}_{}.html".format(os.path.dirname(kwargs["scene"]),
                                                       kwargs["frame"],
                                                       kwargs["figure_specs"]["min"],
                                                       kwargs["figure_specs"]["max"],
                                                       kwargs["figure_specs"]["res"],
                                                       kwargs["figure_specs"]["QX"])
    fig.update_layout(
        title_text=fig_file,
        height=800,
        width=1800)
    fig.show()
    fig.write_html("plots/{}".format(fig_file))


def surface_plot(s, k, e_error, rot_e, tran_e, loss_c, loss_delta, loss_pm, **kwargs):
    loss = eval_loss(c=loss_c, delta=loss_delta, pm=loss_pm, quantile=kwargs["figure_specs"]["QX"])

    print("done")

    fig = make_subplots(
        subplot_titles=("E-error", "Rot-e", "Tran-e", "T-e",
                        "loss C", "loss delta", "loss Omega", "loss"),
        rows=2, cols=4,
        specs=[[{'is_3d': True}, {'is_3d': True}, {'is_3d': True}, {'is_3d': True}],
               [{'is_3d': True}, {'is_3d': True}, {'is_3d': True}, {'is_3d': True}]])

    fig.add_trace(
        go.Surface(x=s, y=k, z=e_error.reshape((len(s), len(s))),
                   colorscale='Viridis',
                   showscale=False,
                   name="e_error"),
        row=1, col=1)

    fig.add_trace(
        go.Surface(x=s, y=k, z=rot_e.reshape(len(s), len(s)),
                   colorscale='Viridis',
                   showscale=False),
        row=1, col=2)

    fig.add_trace(
        go.Surface(x=s, y=k, z=tran_e.reshape(len(s), len(s)),
                   colorscale='Viridis',
                   showscale=False),
        row=1, col=3)

    t_e = rot_e / np.linalg.norm(rot_e) + tran_e / np.linalg.norm(tran_e) + e_error / np.linalg.norm(e_error)
    # / np.linalg.norm(tran_e))
    fig.add_trace(
        go.Surface(x=s, y=k, z=t_e.reshape(len(s), len(s)),
                   colorscale='Viridis',
                   showscale=False),
        row=1, col=4)

    fig.add_trace(
        go.Surface(x=s, y=k, z=loss_c.reshape(len(s), len(s)),
                   colorscale='Viridis',
                   showscale=False),
        row=2, col=1)

    fig.add_trace(
        go.Surface(x=s, y=k, z=loss_delta.reshape(len(s), len(s)),
                   colorscale='Viridis',
                   showscale=False),
        row=2, col=2)

    fig.add_trace(
        go.Surface(x=s, y=k, z=loss_pm.reshape(len(s), len(s)),
                   colorscale='Viridis',
                   showscale=False),
        row=2, col=3)

    fig.add_trace(
        go.Surface(x=s, y=k, z=loss.reshape(len(s), len(s)),
                   colorscale='Viridis',
                   showscale=False),
        row=2, col=4)

    def labels(key):
        return dict(
            xaxis_title='S',
            yaxis_title='K',
            zaxis_title='{}'.format(key))

    fig_file = "surface_{}.{}_{}_{}_{}_{}.html".format(os.path.dirname(kwargs["scene"]),
                                                       kwargs["frame"],
                                                       kwargs["figure_specs"]["min"],
                                                       kwargs["figure_specs"]["max"],
                                                       kwargs["figure_specs"]["res"],
                                                       kwargs["figure_specs"]["QX"])
    fig.update_layout(
        title_text=fig_file,
        height=800,
        width=1800,
        scene1=labels("E-error"),
        scene2=labels("Rot-e"),
        scene3=labels("Tra-e"),
        scene4=labels("Cam-e"),
        scene5=labels("C"),
        scene6=labels("delta"),
        scene7=labels("omega"),
        scene8=labels("Loss"),
    )
    # fig.update_traces(contours_z=dict(show=True, usecolormap=True,
    #                                   highlightcolor="limegreen", project_z=True))
    fig.show()
    fig.write_html("plots/{}".format(fig_file))


def main(res, noise, loc, pts, data_scene, idx_frame, opt_version, **kwargs):
    g8p_norm = norm_8pa(version=opt_version)
    np.random.seed(100)
    # ! Getting a PCL from the dataset
    pcl_dense, pcl_dense_color, _ = data_scene.get_dense_pcl(idx=idx_frame)
    pcl_dense, mask = mask_pcl_by_res_and_loc(pcl=pcl_dense, loc=loc, res=res)
    samples = np.random.randint(0, pcl_dense.shape[1], pts)
    bearings_a, bearings_b, cam_a2b = get_bearings_from_pcl(pcl=pcl_dense[:, samples],
                                                            t_vector=(np.random.uniform(-0.5, 0.5),
                                                                      np.random.uniform(-0.5, 0.5),
                                                                      np.random.uniform(-0.5, 0.5)),
                                                            rot_vector=(np.random.uniform(-10, 10),
                                                                        np.random.uniform(-10, 10),
                                                                        np.random.uniform(-10, 10)),
                                                            noise=noise[0],
                                                            outliers=noise[1] * len(samples))

    plot_pcl_and_cameras(pcl_dense[0:3, samples].T, cam2=cam_a2b)

    e = g8p_norm.build_E_by_cam_pose(cam_a2b)
    print(cam_a2b)
    s = np.linspace(kwargs["figure_specs"]["min"], kwargs["figure_specs"]["max"], kwargs["figure_specs"]["res"])
    k = np.linspace(kwargs["figure_specs"]["min"], kwargs["figure_specs"]["max"], kwargs["figure_specs"]["res"])
    ss, kk = np.meshgrid(s, k)
    e_error = np.zeros_like(kk.flatten())
    rot_error = np.zeros_like(kk.flatten())
    tra_error = np.zeros_like(kk.flatten())
    loss_c = np.zeros_like(kk.flatten())
    loss_delta = np.zeros_like(kk.flatten())
    loss_pm = np.zeros_like(kk.flatten())
    for i in range(ss.size):
        S = ss.flatten()[i]
        K = kk.flatten()[i]
        bearings_a_norm, T1 = g8p_norm.normalizer(bearings_a, S, K)
        bearings_b_norm, T2 = g8p_norm.normalizer(bearings_b, S, K)
        e_hat = g8p_norm.compute_essential_matrix(bearings_a_norm, bearings_b_norm)
        e_hat = np.dot(T1.T, np.dot(e_hat, T2))
        cam_a2b_hat = g8p_norm.recoverPose(e_hat, bearings_a, bearings_b)
        cam_error = evaluate_error_in_transformation(cam_a2b_hat, cam_a2b)
        rot_error[i] = cam_error[0]
        tra_error[i] = cam_error[1]
        x1 = spherical_normalization(bearings_a_norm)
        x2 = spherical_normalization(bearings_b_norm)
        C_2 = get_frobenius_norm(x1, x2)
        C, A = get_frobenius_norm(bearings_a_norm, bearings_b_norm, return_A=True)
        loss_c[i] = C_2
        loss_pm[i] = np.nanmean(angle_between_vectors_arrays(bearings_a_norm, bearings_b_norm))
        u, sigma, v = np.linalg.svd(A)
        loss_delta[i] = sigma[-2]
        e_error[i] = evaluate_error_in_essential_matrix(e, e_hat)
        print(i / ss.size)

    captions = dict(figure_specs=kwargs["figure_specs"],
                    frame=idx_frame,
                    scene=data_scene.scene)
    contour_plot(s, k, msk(e_error, quantile=kwargs["figure_specs"]["QX"]),
                 rot_error, tra_error,
                 msk(loss_c, quantile=kwargs["figure_specs"]["QX"]),
                 loss_delta, loss_pm, **captions)

    surface_plot(s, k, msk(e_error, quantile=kwargs["figure_specs"]["QX"]),
                 rot_error, tra_error,
                 msk(loss_c, quantile=kwargs["figure_specs"]["QX"]),
                 loss_delta, loss_pm, **captions)


if __name__ == '__main__':
    path = "/home/kike/Documents/datasets/MP3D_VO"
    # path = "/run/user/1001/gvfs/sftp:host=140.114.27.95,port=50002/NFS/kike/minos/vslab_MP3D_VO/512x1024"
    data = MP3D_VO(scene="2azQ1b91cZZ/0", path=path)

    res = (66, 46)
    figure_specs_ = dict(
        min=0.01,
        max=10,
        res=20,
        QX=0.1
    )
    main(res=res,
         noise=(500, 0.05),
         loc=(0, 0),
         pts=200,
         data_scene=data,
         idx_frame=150,
         opt_version="v2",
         figure_specs=figure_specs_)
