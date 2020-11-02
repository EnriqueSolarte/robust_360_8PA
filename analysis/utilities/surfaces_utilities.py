from analysis.utilities.plot_utilities import *


def create_grid(**kwargs):
    v = np.linspace(
        start=kwargs["grid"][0], stop=kwargs["grid"][1], num=kwargs["grid"][2])
    ss, kk = np.meshgrid(v, v)

    kwargs["v_grid"] = v
    kwargs["ss_grid"] = ss.flatten()
    kwargs["kk_grid"] = kk.flatten()
    kwargs["grid_range"] = range(len(v) * len(v))
    kwargs["surfaces"] = dict()
    return kwargs


def add_surfaces_evaluation(label, value, **kwargs):
    if not label in kwargs["surfaces"].keys():
        kwargs["surfaces"][label] = [value]
    else:
        kwargs["surfaces"][label].append(value)
    return kwargs


def plot_surfaces(**kwargs):
    surfaces_names = sorted(list(kwargs["surfaces"].keys()))

    fig = make_subplots(
        subplot_titles=surfaces_names,
        rows=2,
        cols=3,
        specs=[[{
            'is_3d': True
        }, {
            'is_3d': True
        }, {
            'is_3d': True
        }], [{
            'is_3d': True
        }, {
            'is_3d': True
        }, {
            'is_3d': True
        }]])
    idxs = np.linspace(0, 5, 6).reshape(2, -1)
    for i, eval in enumerate(surfaces_names):
        loc = np.squeeze(np.where(idxs == i))
        surface = np.array(kwargs["surfaces"][eval])
        fig.add_trace(
            go.Surface(
                x=kwargs["v_grid"],
                y=kwargs["v_grid"],
                z=surface.reshape((len(kwargs["v_grid"]), len(
                    kwargs["v_grid"]))),
                colorscale='Viridis',
                showscale=False),
            row=loc[0] + 1,
            col=loc[1] + 1)
        fig.add_trace(
            go.Scatter3d(
                x=(1,),
                y=(1,),
                z=(kwargs["8PA"][eval],),
                marker=dict(color=colors["COLOR_GENERAL"], size=5),
                name="8PA"),
            row=loc[0] + 1,
            col=loc[1] + 1)

    fig_file = "{}.html".format(kwargs["filename"])
    fig.update_layout(title_text=fig_file, height=800, width=1800)
    fig.show()
    fig.write_html("{}".format(fig_file))


def get_eval_of_8PA(**kwargs):
    kwargs["8PA"] = dict()
    kwargs["8PA"]["cam_pose"], _ = get_cam_pose_by_8pa(**kwargs)

    kwargs["8PA"]["e"] = g8p().compute_essential_matrix(
        x1=kwargs["bearings"]["kf"], x2=kwargs["bearings"]["frm"])

    kwargs["8PA"]["e_error"] = evaluate_error_in_essential_matrix(
        e_ref=kwargs["e_gt"], e_hat=kwargs["8PA"]["e"])

    residuals = projected_error(
        e=kwargs["8PA"]["e"],
        x1=kwargs["bearings"]["kf"],
        x2=kwargs["bearings"]["frm"])

    residuals_norm = algebraic_error(
        e=kwargs["8PA"]["e"],
        x1=kwargs["bearings"]["kf"],
        x2=kwargs["bearings"]["frm"])

    kwargs["8PA"]["residuals_norm_error"] = np.sum(residuals_norm**2)
    kwargs["8PA"]["residuals_error"] = np.sum(residuals**2)

    landmarks_hat = np.linalg.inv(
        kwargs["8PA"]["cam_pose"]) @ kwargs["landmarks_kf"]
    reprojection = get_projection_error_between_vectors_arrays(
        array_ref=kwargs["bearings"]["frm"],
        array_vector=landmarks_hat[0:3, :])
    kwargs["8PA"]["reprojection_error"] = np.sum(1 / reprojection)
    cam_error = evaluate_error_in_transformation(
        transform_gt=kwargs["cam_gt"], transform_est=kwargs["8PA"]["cam_pose"])
    kwargs["8PA"]["rot_error"] = cam_error[0]
    kwargs["8PA"]["tran_error"] = cam_error[1]
    return kwargs


def eval_surfaces(**kwargs):
    cam_error = evaluate_error_in_transformation(
        transform_gt=kwargs["cam_gt"], transform_est=kwargs["cam_hat"])
    rot_error, tran_error = cam_error[0], cam_error[1]
    kwargs = add_surfaces_evaluation(
        label="rot_error", value=rot_error, **kwargs)
    kwargs = add_surfaces_evaluation(
        label="tran_error", value=tran_error, **kwargs)

    e_error = evaluate_error_in_essential_matrix(
        e_ref=kwargs["e_gt"], e_hat=kwargs["e_hat"])
    kwargs = add_surfaces_evaluation(
        label="e_error", value=e_error, **kwargs)

    # residuals_norm_error = projected_distance(
    #     e=kwargs["e_norm"],
    #     x1=kwargs["bearings"]["kf_norm"],
    #     x2=kwargs["bearings"]["frm_norm"],
    # )
    residuals_norm_error = algebraic_error(
        e=kwargs["e_norm"],
        x1=kwargs["bearings"]["kf_norm"],
        x2=kwargs["bearings"]["frm_norm"],
    )
    kwargs = add_surfaces_evaluation(
        label="residuals_norm_error",
        value=np.sum(residuals_norm_error ** 2),
        **kwargs)

    residuals_error = projected_error(
        e=kwargs["e_hat"],
        x1=kwargs["bearings"]["kf"],
        x2=kwargs["bearings"]["frm"],
    )
    kwargs = add_surfaces_evaluation(
        label="residuals_error", value=np.sum(residuals_error ** 2), **kwargs)

    landmark_hat = np.linalg.inv(kwargs["cam_hat"]) @ kwargs["landmarks_kf"]
    reprojection_error = get_projection_error_between_vectors_arrays(
        array_ref=kwargs["bearings"]["frm"], array_vector=landmark_hat[0:3, :])
    kwargs = add_surfaces_evaluation(
        label="reprojection_error",
        value=np.sum(1 / reprojection_error),
        **kwargs)
    return kwargs
