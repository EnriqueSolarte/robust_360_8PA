from config import *
import plotly.graph_objects as go
import numpy as np
import pandas as pd


def plot_errors(noise, res, point, save=True):
    _ours_m, _8pa_m, _ours_v, _8pa_v = [], [], [], []
    x = []

    if experiment_group == "noise":
        x = noises[::-1]
        for noise in noises[::-1]:
            dt = pd.read_csv(
                "../report/{}/{}/{}/{}/{}/{}/{}/{}_{}_{}_{}_{}_{}_{}_{}.csv".
                format(dataset, scene, str(idx_frame),
                       "mc" if motion_constraint else "!mc", noise,
                       str(res[0]) + "x" + str(res[1]), point, scene[:-2],
                       scene[-1:], str(idx_frame),
                       "mc" if motion_constraint else "!mc", noise,
                       str(res[0]) + "x" + str(res[1]), point, opt_version))
            data = dt.values
            # ! Ours' method
            _ours_m.append(np.median(data[:, 2:4], axis=0))
            _ours_v.append(np.var(data[:, 2:4], axis=0))
            # ! 8PA
            _8pa_m.append(np.median(data[:, 0:2], axis=0))
            _8pa_v.append(np.var(data[:, 0:2], axis=0))

    elif experiment_group == "fov":
        x = ress
        for res in ress:
            dt = pd.read_csv(
                "../report/{}/{}/{}/{}/{}/{}/{}/{}_{}_{}_{}_{}_{}_{}_{}.csv".
                format(dataset, scene, str(idx_frame),
                       "mc" if motion_constraint else "!mc", noise,
                       str(res[0]) + "x" + str(res[1]), point, scene[:-2],
                       scene[-1:], str(idx_frame),
                       "mc" if motion_constraint else "!mc", noise,
                       str(res[0]) + "x" + str(res[1]), point, opt_version))
            data = dt.values
            # ! Ours' method
            _ours_m.append(np.median(data[:, 2:4], axis=0))
            _ours_v.append(np.var(data[:, 2:4], axis=0))
            # ! 8PA
            _8pa_m.append(np.median(data[:, 0:2], axis=0))
            _8pa_v.append(np.var(data[:, 0:2], axis=0))

    elif experiment_group == "point":
        x = points
        for point in points:
            dt = pd.read_csv(
                "../report/{}/{}/{}/{}/{}/{}/{}/{}_{}_{}_{}_{}_{}_{}_{}.csv".
                format(dataset, scene, str(idx_frame),
                       "mc" if motion_constraint else "!mc", noise,
                       str(res[0]) + "x" + str(res[1]), point, scene[:-2],
                       scene[-1:], str(idx_frame),
                       "mc" if motion_constraint else "!mc", noise,
                       str(res[0]) + "x" + str(res[1]), point, opt_version))
            data = dt.values
            # ! Ours' method
            _ours_m.append(np.median(data[:, 2:4], axis=0))
            _ours_v.append(np.var(data[:, 2:4], axis=0))
            # ! 8PA
            _8pa_m.append(np.median(data[:, 0:2], axis=0))
            _8pa_v.append(np.var(data[:, 0:2], axis=0))

    _ours_m = np.array(_ours_m)
    _8pa_m = np.array(_8pa_m)
    _ours_v = np.array(_ours_v)
    _8pa_v = np.array(_8pa_v)

    fig = go.Figure()

    # ! Ours method
    fig.add_trace(go.Scatter(
        x=x,
        y=_ours_m[:, 0],
        name='ours-rot',
    ))
    fig.add_trace(go.Scatter(
        x=x,
        y=_ours_m[:, 1],
        name='ours-trans',
    ))

    # ! 8PA
    fig.add_trace(go.Scatter(
        x=x,
        y=_8pa_m[:, 0],
        name='8pa-rot',
    ))
    fig.add_trace(go.Scatter(
        x=x,
        y=_8pa_m[:, 1],
        name='8pa-trans',
    ))

    fig.update_layout(title="{}_{}_{}_{}_{}_{}_{}_{}_{}_{}".format(
        dataset, scene[:-2], scene[-1:], str(idx_frame),
        "mc" if motion_constraint else "!mc", experiment_group,
        noise if experiment_group != "noise" else "",
        str(res[0]) + "x" + str(res[1]) if experiment_group != "fov" else "",
        point if experiment_group != "point" else "", opt_version),
                      xaxis_title=experiment_group[0].upper() +
                      experiment_group[1:],
                      yaxis_title="Error",
                      font=dict(family="Courier New, monospace",
                                size=16,
                                color="RebeccaPurple"))
    fig.update_traces(mode='lines')

    if save:
        # ! Save .html
        fig.write_html(
            "../report/{}/{}/{}/{}_{}_{}_{}_{}_{}_{}_{}_{}.html".format(
                dataset, scene, str(idx_frame), scene[:-2], scene[-1:],
                str(idx_frame), "mc" if motion_constraint else "!mc",
                experiment_group, noise if experiment_group != "noise" else "",
                str(res[0]) + "x" +
                str(res[1]) if experiment_group != "fov" else "",
                point if experiment_group != "point" else "", opt_version))

        # ! Save .png
        fig.write_image(
            "../report/{}/{}/{}/{}_{}_{}_{}_{}_{}_{}_{}_{}.png".format(
                dataset, scene, str(idx_frame), scene[:-2], scene[-1:],
                str(idx_frame), "mc" if motion_constraint else "!mc",
                experiment_group, noise if experiment_group != "noise" else "",
                str(res[0]) + "x" +
                str(res[1]) if experiment_group != "fov" else "",
                point if experiment_group != "point" else "", opt_version))

    fig.show()


if __name__ == "__main__":
    plot_errors(noise, res, point, save=True)
