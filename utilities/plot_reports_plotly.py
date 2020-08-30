import sys
import os

sys.path.append(os.getcwd())

from config import *
import plotly.graph_objects as go
import numpy as np
import pandas as pd

from plotly.subplots import make_subplots


def plot_point_distribution(matches):
    import plotly.express as px
    fig = px.scatter(x=matches[0, :, 0], y=matches[0, :, 1])
    fig.show()


def plot_sk_values(noise, res, point, save=False):
    s1_m, s1_std, s2_m, s2_std = [], [], [], []
    k1_m, k1_std, k2_m, k2_std = [], [], [], []

    if experiment_group == "noise":
        x = noises[::-1]
        for noise in noises[::-1]:
            dt = pd.read_csv(
                output_dir +
                "/{}/{}/{}/{}/{}/{}/{}/{}/{}_{}_{}_{}_{}_{}_{}_{}.csv".format(
                    experiment, dataset, scene, str(idx_frame),
                    "mc" if motion_constraint else "!mc", noise,
                    str(res[0]) + "x" +
                    str(res[1]), point, scene[:-2], scene[-1:], str(idx_frame),
                    "mc" if motion_constraint else "!mc", noise,
                    str(res[0]) + "x" + str(res[1]), point, opt_version))
            data = dt.values
            # ! S and K values
            s1_m.append(np.median(data[:, 4]))
            s1_std.append(1.5 * np.std(data[:, 4]))
            k1_m.append(np.median(data[:, 5]))
            k1_std.append(1.5 * np.std(data[:, 5]))

            if opt_version != "v1":
                s2_m.append(np.median(data[:, 6]))
                s2_std.append(1.5 * np.std(data[:, 6]))
                k2_m.append(np.median(data[:, 7]))
                k2_std.append(1.5 * np.std(data[:, 7]))

    elif experiment_group == "fov":
        x = list(range(len(ress)))
        for res in ress:
            dt = pd.read_csv(
                output_dir +
                "/{}/{}/{}/{}/{}/{}/{}/{}/{}_{}_{}_{}_{}_{}_{}_{}.csv".format(
                    experiment, dataset, scene, str(idx_frame),
                    "mc" if motion_constraint else "!mc", noise,
                    str(res[0]) + "x" +
                    str(res[1]), point, scene[:-2], scene[-1:], str(idx_frame),
                    "mc" if motion_constraint else "!mc", noise,
                    str(res[0]) + "x" + str(res[1]), point, opt_version))
            data = dt.values
            # ! S and K values
            s1_m.append(np.median(data[:, 4]))
            s1_std.append(1.5 * np.std(data[:, 4]))
            k1_m.append(np.median(data[:, 5]))
            k1_std.append(1.5 * np.std(data[:, 5]))

            if opt_version != "v1":
                s2_m.append(np.median(data[:, 6]))
                s2_std.append(1.5 * np.std(data[:, 6]))
                k2_m.append(np.median(data[:, 7]))
                k2_std.append(1.5 * np.std(data[:, 7]))

    elif experiment_group == "point":
        x = points
        for point in points:
            dt = pd.read_csv(
                output_dir +
                "/{}/{}/{}/{}/{}/{}/{}/{}/{}_{}_{}_{}_{}_{}_{}_{}.csv".format(
                    experiment, dataset, scene, str(idx_frame),
                    "mc" if motion_constraint else "!mc", noise,
                    str(res[0]) + "x" +
                    str(res[1]), point, scene[:-2], scene[-1:], str(idx_frame),
                    "mc" if motion_constraint else "!mc", noise,
                    str(res[0]) + "x" + str(res[1]), point, opt_version))
            data = dt.values
            # ! S and K values
            s1_m.append(np.median(data[:, 4]))
            s1_std.append(1.5 * np.std(data[:, 4]))
            k1_m.append(np.median(data[:, 5]))
            k1_std.append(1.5 * np.std(data[:, 5]))

            if opt_version != "v1":
                s2_m.append(np.median(data[:, 6]))
                s2_std.append(1.5 * np.std(data[:, 6]))
                k2_m.append(np.median(data[:, 7]))
                k2_std.append(1.5 * np.std(data[:, 7]))

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(x=x,
                   y=s1_m,
                   mode='markers',
                   error_y=dict(type='data', array=s1_std),
                   name='s1'))
    fig.add_trace(
        go.Scatter(x=x,
                   y=k1_m,
                   mode='markers',
                   error_y=dict(type='data', array=k1_std),
                   name='k1'))

    # s1_sdw = [s1_m[i] + s1_std[i] for i in range(len(s1_m))] + [s1_m[i] - s1_std[i] for i in range(len(s1_m))]
    # fig.add_trace(go.Scatter(
    #     x=x,
    #     y=s1_sdw,
    #     fill='toself',
    #     showlegend=False,
    #     name='Ideal',
    # ))

    if experiment_group == "fov":
        fig.update_xaxes(
            ticktext=[str(res[0]) + 'x' + str(res[1]) for res in ress],
            tickvals=x,
        )

    fig.update_layout(title="{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_s1k1".format(
        experiment, dataset, scene[:-2], scene[-1:], str(idx_frame),
        "mc" if motion_constraint else "!mc", experiment_group,
        noise if experiment_group != "noise" else "",
        str(res[0]) + "x" + str(res[1]) if experiment_group != "fov" else "",
        point if experiment_group != "point" else "", opt_version),
                      xaxis_title=experiment_group[0].upper() +
                      experiment_group[1:],
                      yaxis_title="Error",
                      font=dict(family="Courier New, monospace", size=14))

    # fig.update_layout(showlegend=False)

    # Save *s1k1.png
    if save:
        # ! Save .png
        fig.write_image(
            output_dir +
            "/{}/{}/{}/{}/{}_{}_{}_{}_{}_{}_{}_{}_{}_s1k1.png".format(
                experiment,
                dataset, scene, str(idx_frame), scene[:-2], scene[-1:],
                str(idx_frame), "mc" if motion_constraint else "!mc",
                experiment_group, noise if experiment_group != "noise" else "",
                str(res[0]) + "x" +
                str(res[1]) if experiment_group != "fov" else "",
                point if experiment_group != "point" else "", opt_version),
            scale=2)
        '''
        # ! Save .html
        fig.write_html(
            output_dir + "/{}/{}/{}/{}/{}_{}_{}_{}_{}_{}_{}_{}_{}.html".format(
                experiment, dataset, scene, str(idx_frame), scene[:-2], scene[-1:],
                str(idx_frame), "mc"
                if motion_constraint else "!mc", experiment_group, noise
                if experiment_group != "noise" else "",
                str(res[0]) + "x" + str(res[1])
                if experiment_group != "fov" else "", point
                if experiment_group != "point" else "", opt_version))
        
        # ! Save .svg
        fig.write_image(
            output_dir + "/{}/{}/{}/{}/{}_{}_{}_{}_{}_{}_{}_{}_{}.svg".format(
                experiment, dataset, scene, str(idx_frame), scene[:-2], scene[-1:],
                str(idx_frame), "mc"
                if motion_constraint else "!mc", experiment_group, noise
                if experiment_group != "noise" else "",
                str(res[0]) + "x" + str(res[1])
                if experiment_group != "fov" else "", point
                if experiment_group != "point" else "", opt_version))
        '''

    fig.show()

    if opt_version != "v1":
        fig = go.Figure()

        fig.add_trace(
            go.Scatter(x=x,
                       y=s2_m,
                       mode='markers',
                       error_y=dict(type='data', array=s2_std),
                       name='s2'))
        fig.add_trace(
            go.Scatter(x=x,
                       y=k2_m,
                       mode='markers',
                       error_y=dict(type='data', array=k2_std),
                       name='k2'))

        if experiment_group == "fov":
            fig.update_xaxes(
                ticktext=[str(res[0]) + 'x' + str(res[1]) for res in ress],
                tickvals=x,
            )

        fig.update_layout(title="{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_s2k2".format(
            experiment, dataset, scene[:-2], scene[-1:], str(idx_frame),
            "mc" if motion_constraint else "!mc", experiment_group,
            noise if experiment_group != "noise" else "",
            str(res[0]) + "x" +
            str(res[1]) if experiment_group != "fov" else "",
            point if experiment_group != "point" else "", opt_version),
                          xaxis_title=experiment_group[0].upper() +
                          experiment_group[1:],
                          yaxis_title="Error",
                          font=dict(family="Courier New, monospace", size=14))

        # fig.update_layout(showlegend=False)

        # Save *s2k2.png
        if save:
            # ! Save .png
            fig.write_image(
                output_dir +
                "/{}/{}/{}/{}/{}_{}_{}_{}_{}_{}_{}_{}_{}_s2k2.png".format(
                    experiment, dataset, scene, str(idx_frame), scene[:-2],
                    scene[-1:], str(idx_frame),
                    "mc" if motion_constraint else "!mc", experiment_group,
                    noise if experiment_group != "noise" else "",
                    str(res[0]) + "x" +
                    str(res[1]) if experiment_group != "fov" else "",
                    point if experiment_group != "point" else "", opt_version),
                scale=2)
            '''
            # ! Save .html
            fig.write_html(
                output_dir + "/{}/{}/{}/{}/{}_{}_{}_{}_{}_{}_{}_{}_{}.html".format(
                    experiment, dataset, scene, str(idx_frame), scene[:-2], scene[-1:],
                    str(idx_frame), "mc"
                    if motion_constraint else "!mc", experiment_group, noise
                    if experiment_group != "noise" else "",
                    str(res[0]) + "x" + str(res[1])
                    if experiment_group != "fov" else "", point
                    if experiment_group != "point" else "", opt_version))

            # ! Save .svg
            fig.write_image(
                output_dir + "/{}/{}/{}/{}/{}_{}_{}_{}_{}_{}_{}_{}_{}.svg".format(
                    experiment, dataset, scene, str(idx_frame), scene[:-2], scene[-1:],
                    str(idx_frame), "mc"
                    if motion_constraint else "!mc", experiment_group, noise
                    if experiment_group != "noise" else "",
                    str(res[0]) + "x" + str(res[1])
                    if experiment_group != "fov" else "", point
                    if experiment_group != "point" else "", opt_version))
            '''

        fig.show()


def plot_errors(noise, res, point, save=True):
    _ours_m, _8pa_m, _ours_std, _8pa_std = [], [], [], []
    x = []

    if experiment_group == "noise":
        x = noises[::-1]
        for noise in noises[::-1]:
            dt = pd.read_csv(
                output_dir +
                "/{}/{}/{}/{}/{}/{}/{}/{}/{}_{}_{}_{}_{}_{}_{}_{}.csv".format(
                    experiment, dataset, scene, str(idx_frame),
                    "mc" if motion_constraint else "!mc", noise,
                    str(res[0]) + "x" +
                    str(res[1]), point, scene[:-2], scene[-1:], str(idx_frame),
                    "mc" if motion_constraint else "!mc", noise,
                    str(res[0]) + "x" + str(res[1]), point, opt_version))
            data = dt.values
            # ! Ours' method
            _ours_m.append(np.median(data[:, 2:4], axis=0))
            _ours_std.append(1.5 * np.std(data[:, 2:4], axis=0))
            # ! 8PA
            _8pa_m.append(np.median(data[:, 0:2], axis=0))
            _8pa_std.append(1.5 * np.std(data[:, 0:2], axis=0))

    elif experiment_group == "fov":
        x = list(range(len(ress)))
        for res in ress:
            dt = pd.read_csv(
                output_dir +
                "/{}/{}/{}/{}/{}/{}/{}/{}/{}_{}_{}_{}_{}_{}_{}_{}.csv".format(
                    experiment, dataset, scene, str(idx_frame),
                    "mc" if motion_constraint else "!mc", noise,
                    str(res[0]) + "x" +
                    str(res[1]), point, scene[:-2], scene[-1:], str(idx_frame),
                    "mc" if motion_constraint else "!mc", noise,
                    str(res[0]) + "x" + str(res[1]), point, opt_version))
            data = dt.values
            # ! Ours' method
            _ours_m.append(np.median(data[:, 2:4], axis=0))
            _ours_std.append(1.5 * np.std(data[:, 2:4], axis=0))
            # ! 8PA
            _8pa_m.append(np.median(data[:, 0:2], axis=0))
            _8pa_std.append(1.5 * np.std(data[:, 0:2], axis=0))

    elif experiment_group == "point":
        x = points
        for point in points:
            dt = pd.read_csv(
                output_dir +
                "/{}/{}/{}/{}/{}/{}/{}/{}/{}_{}_{}_{}_{}_{}_{}_{}.csv".format(
                    experiment, dataset, scene, str(idx_frame),
                    "mc" if motion_constraint else "!mc", noise,
                    str(res[0]) + "x" +
                    str(res[1]), point, scene[:-2], scene[-1:], str(idx_frame),
                    "mc" if motion_constraint else "!mc", noise,
                    str(res[0]) + "x" + str(res[1]), point, opt_version))
            data = dt.values
            # ! Ours' method
            _ours_m.append(np.median(data[:, 2:4], axis=0))
            _ours_std.append(1.5 * np.std(data[:, 2:4], axis=0))
            # ! 8PA
            _8pa_m.append(np.median(data[:, 0:2], axis=0))
            _8pa_std.append(1.5 * np.std(data[:, 0:2], axis=0))

    _ours_m = np.array(_ours_m)
    _8pa_m = np.array(_8pa_m)
    _ours_std = np.array(_ours_std)
    _8pa_std = np.array(_8pa_std)

    _ours_sdw = [_ours_m[i] + _ours_std[i] for i in range(len(_ours_m))] + \
                [_ours_m[i] - _ours_std[i] for i in range(len(_ours_m))]

    _8pa_sdw = [_8pa_m[i] + _8pa_std[i] for i in range(len(_8pa_m))] + \
               [_8pa_m[i] - _8pa_std[i] for i in range(len(_8pa_m))]

    _ours_sdw = np.array(_ours_sdw)
    _8pa_sdw = np.array(_8pa_sdw)

    fig = go.Figure()
    fig = make_subplots(rows=1, cols=2)

    # ! Ours method
    fig.add_trace(
        go.Scatter(
            x=x,
            y=_ours_m[:, 0],
            name='ours-rot',
            # error_y=dict(type='data', array=_ours_std[:, 0])
        ),
        row=1,
        col=1)
    fig.add_trace(
        go.Scatter(
            x=x,
            y=_ours_m[:, 1],
            name='ours-trans',
            # error_y=dict(type='data', array=_ours_std[:, 1])
        ),
        row=1,
        col=2)

    # ! 8PA
    fig.add_trace(
        go.Scatter(
            x=x,
            y=_8pa_m[:, 0],
            name='8pa-rot',
            line=dict(width=2, dash='dot'),
            # error_y=dict(type='data', array=_8pa_std[:, 0])
        ),
        row=1,
        col=1)
    fig.add_trace(
        go.Scatter(
            x=x,
            y=_8pa_m[:, 1],
            name='8pa-trans',
            line=dict(width=2, dash='dot'),
            # error_y=dict(type='data', array=_8pa_std[:, 1])
        ),
        row=1,
        col=2)

    _ours_up = _ours_sdw[0:len(x)]
    _ours_low = _ours_sdw[len(x) * 2:len(x) - 1:-1]
    _8pa_up = _8pa_sdw[0:len(x)]
    _8pa_low = _8pa_sdw[len(x) * 2:len(x) - 1:-1]

    fig.add_trace(go.Scatter(x=x + x[::-1],
                             y=_ours_up[:, 0].tolist() +
                             _ours_low[:, 0].tolist(),
                             fill='toself',
                             fillcolor='rgba(0,0,255,0.1)',
                             line_color='rgba(255,255,255,0)',
                             name='ours-rot'),
                  row=1,
                  col=1)

    fig.add_trace(go.Scatter(x=x + x[::-1],
                             y=_ours_up[:, 1].tolist() +
                             _ours_low[:, 1].tolist(),
                             fill='toself',
                             fillcolor='rgba(255,0,0,0.1)',
                             line_color='rgba(255,255,255,0)',
                             name='ours-trans'),
                  row=1,
                  col=2)

    fig.add_trace(go.Scatter(
        x=x + x[::-1],
        y=_8pa_up[:, 0].tolist() + _8pa_low[:, 0].tolist(),
        fill='toself',
        fillcolor='rgba(0,204,150,0.1)',
        line_color='rgba(255,255,255,0)',
        name='8pa-rot',
    ),
                  row=1,
                  col=1)

    fig.add_trace(go.Scatter(
        x=x + x[::-1],
        y=_8pa_up[:, 1].tolist() + _8pa_low[:, 1].tolist(),
        fill='toself',
        fillcolor='rgba(171,99,250,0.1)',
        line_color='rgba(255,255,255,0)',
        name='8pa-trans',
    ),
                  row=1,
                  col=2)

    fig.update_layout(
        title="{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}".format(
            experiment, dataset, scene[:-2], scene[-1:], str(idx_frame),
            "mc" if motion_constraint else "!mc", experiment_group,
            noise if experiment_group != "noise" else "",
            str(res[0]) + "x" +
            str(res[1]) if experiment_group != "fov" else "",
            point if experiment_group != "point" else "", opt_version),
        # xaxis_title=experiment_group[0].upper() +
        # experiment_group[1:],
        # yaxis_title="Error",
        font=dict(
            family="Courier New, monospace",
            size=14,
        ))
    # color = "RebeccaPurple"
    # , xaxis_type="log"

    fig.update_traces(mode='lines+markers', line_shape='linear')
    fig.update_xaxes(title_text=experiment_group[0].upper() +
                     experiment_group[1:] + " - Rot",
                     row=1,
                     col=1)
    fig.update_xaxes(title_text=experiment_group[0].upper() +
                     experiment_group[1:] + " - Trans",
                     row=1,
                     col=2)

    fig.update_yaxes(title_text="Error", row=1, col=1)
    fig.update_yaxes(title_text="Error", row=1, col=2)
    # fig.update_traces(mode='lines', line_shape='spline')
    fig.update_layout(showlegend=False)

    if experiment_group == "fov":
        fig.update_xaxes(
            ticktext=[str(res[0]) + 'x' + str(res[1]) for res in ress],
            tickvals=x,
        )

    fig.show()
    if save:
        # ! Save .png
        fig.update_layout(width=1000, height=500)
        fig.write_image(
            output_dir + "/{}/{}/{}/{}/{}_{}_{}_{}_{}_{}_{}_{}_{}.png".format(
                experiment,
                dataset, scene, str(idx_frame), scene[:-2], scene[-1:],
                str(idx_frame), "mc" if motion_constraint else "!mc",
                experiment_group, noise if experiment_group != "noise" else "",
                str(res[0]) + "x" +
                str(res[1]) if experiment_group != "fov" else "",
                point if experiment_group != "point" else "", opt_version),
            scale=2)
        '''
        # ! Save .html
        fig.write_html(
            output_dir + "/{}/{}/{}/{}/{}_{}_{}_{}_{}_{}_{}_{}_{}.html".format(
                experiment, dataset, scene, str(idx_frame), scene[:-2], scene[-1:],
                str(idx_frame), "mc"
                if motion_constraint else "!mc", experiment_group, noise
                if experiment_group != "noise" else "",
                str(res[0]) + "x" + str(res[1])
                if experiment_group != "fov" else "", point
                if experiment_group != "point" else "", opt_version))

        # ! Save .svg
        fig.write_image(
            output_dir + "/{}/{}/{}/{}/{}_{}_{}_{}_{}_{}_{}_{}_{}.svg".format(
                experiment, dataset, scene, str(idx_frame), scene[:-2], scene[-1:],
                str(idx_frame), "mc"
                if motion_constraint else "!mc", experiment_group, noise
                if experiment_group != "noise" else "",
                str(res[0]) + "x" + str(res[1])
                if experiment_group != "fov" else "", point
                if experiment_group != "point" else "", opt_version))
        '''


if __name__ == "__main__":
    plot_errors(noise, res, point, save=True)
    if experiment_group == experiment_group_choices[1]:
        plot_sk_values(noise, res, point, save=True)
