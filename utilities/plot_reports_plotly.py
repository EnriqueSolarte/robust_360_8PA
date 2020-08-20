import sys
import os

sys.path.append(os.getcwd())

from config import *
import plotly.graph_objects as go
import numpy as np
import pandas as pd


def plot_sk_values(noise, res, point, save=False):
    s1_m, s1_std, s2_m, s2_std = [], [], [], []
    k1_m, k1_std, k2_m, k2_std = [], [], [], []

    if experiment_group == "noise":
        x = noises[::-1]
        for noise in noises[::-1]:
            dt = pd.read_csv(
                "../report/{}/{}/{}/{}/{}/{}/{}/{}_{}_{}_{}_{}_{}_{}_{}.csv".
                    format(dataset, scene, str(idx_frame), "mc"
                if motion_constraint else "!mc", noise,
                           str(res[0]) + "x" + str(res[1]), point, scene[:-2],
                           scene[-1:], str(idx_frame), "mc"
                           if motion_constraint else "!mc", noise,
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
                "../report/{}/{}/{}/{}/{}/{}/{}/{}_{}_{}_{}_{}_{}_{}_{}.csv".
                    format(dataset, scene, str(idx_frame), "mc"
                if motion_constraint else "!mc", noise,
                           str(res[0]) + "x" + str(res[1]), point, scene[:-2],
                           scene[-1:], str(idx_frame), "mc"
                           if motion_constraint else "!mc", noise,
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
                "../report/{}/{}/{}/{}/{}/{}/{}/{}_{}_{}_{}_{}_{}_{}_{}.csv".
                    format(dataset, scene, str(idx_frame), "mc"
                if motion_constraint else "!mc", noise,
                           str(res[0]) + "x" + str(res[1]), point, scene[:-2],
                           scene[-1:], str(idx_frame), "mc"
                           if motion_constraint else "!mc", noise,
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

    fig.add_trace(go.Scatter(x=x, y=s1_m, mode='markers', error_y=dict(
        type='data', array=s1_std), name='s1'))
    fig.add_trace(go.Scatter(x=x, y=k1_m, mode='markers', error_y=dict(
        type='data', array=k1_std), name='k1'))

    if experiment_group == "fov":
        fig.update_xaxes(
            ticktext=[str(res[0]) + 'x' + str(res[1]) for res in ress],
            tickvals=x,
        )

    fig.update_layout(
        title="{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_s1k1".format(
            dataset, scene[:-2], scene[-1:], str(idx_frame), "mc"
            if motion_constraint else "!mc", experiment_group, noise
            if experiment_group != "noise" else "",
            str(res[0]) + "x" + str(res[1])
            if experiment_group != "fov" else "", point
            if experiment_group != "point" else "", opt_version),
        xaxis_title=experiment_group[0].upper() + experiment_group[1:],
        yaxis_title="Error",
        font=dict(
            family="Courier New, monospace", size=14, color="RebeccaPurple"))

    # fig.update_layout(showlegend=False)

    # Save *s1k1.png
    if save:
        # ! Save .png
        fig.write_image(
            "../report/{}/{}/{}/{}_{}_{}_{}_{}_{}_{}_{}_{}_s1k1.png".format(
                dataset, scene, str(idx_frame), scene[:-2], scene[-1:],
                str(idx_frame), "mc"
                if motion_constraint else "!mc", experiment_group, noise
                if experiment_group != "noise" else "",
                str(res[0]) + "x" + str(res[1])
                if experiment_group != "fov" else "", point
                if experiment_group != "point" else "", opt_version), scale=2)

        '''
        # ! Save .html
        fig.write_html(
            "../report/{}/{}/{}/{}_{}_{}_{}_{}_{}_{}_{}_{}.html".format(
                dataset, scene, str(idx_frame), scene[:-2], scene[-1:],
                str(idx_frame), "mc"
                if motion_constraint else "!mc", experiment_group, noise
                if experiment_group != "noise" else "",
                str(res[0]) + "x" + str(res[1])
                if experiment_group != "fov" else "", point
                if experiment_group != "point" else "", opt_version))
        
        # ! Save .svg
        fig.write_image(
            "../report/{}/{}/{}/{}_{}_{}_{}_{}_{}_{}_{}_{}.svg".format(
                dataset, scene, str(idx_frame), scene[:-2], scene[-1:],
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

        fig.add_trace(go.Scatter(x=x, y=s2_m, mode='markers', error_y=dict(
            type='data', array=s2_std), name='s2'))
        fig.add_trace(go.Scatter(x=x, y=k2_m, mode='markers', error_y=dict(
            type='data', array=k2_std), name='k2'))

        if experiment_group == "fov":
            fig.update_xaxes(
                ticktext=[str(res[0]) + 'x' + str(res[1]) for res in ress],
                tickvals=x,
            )

        fig.update_layout(
            title="{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_s2k2".format(
                dataset, scene[:-2], scene[-1:], str(idx_frame), "mc"
                if motion_constraint else "!mc", experiment_group, noise
                if experiment_group != "noise" else "",
                str(res[0]) + "x" + str(res[1])
                if experiment_group != "fov" else "", point
                if experiment_group != "point" else "", opt_version),
            xaxis_title=experiment_group[0].upper() + experiment_group[1:],
            yaxis_title="Error",
            font=dict(
                family="Courier New, monospace", size=14, color="RebeccaPurple"))

        # fig.update_layout(showlegend=False)

        # Save *s2k2.png
        if save:
            # ! Save .png
            fig.write_image(
                "../report/{}/{}/{}/{}_{}_{}_{}_{}_{}_{}_{}_{}_s2k2.png".format(
                    dataset, scene, str(idx_frame), scene[:-2], scene[-1:],
                    str(idx_frame), "mc"
                    if motion_constraint else "!mc", experiment_group, noise
                    if experiment_group != "noise" else "",
                    str(res[0]) + "x" + str(res[1])
                    if experiment_group != "fov" else "", point
                    if experiment_group != "point" else "", opt_version), scale=2)

            '''
            # ! Save .html
            fig.write_html(
                "../report/{}/{}/{}/{}_{}_{}_{}_{}_{}_{}_{}_{}.html".format(
                    dataset, scene, str(idx_frame), scene[:-2], scene[-1:],
                    str(idx_frame), "mc"
                    if motion_constraint else "!mc", experiment_group, noise
                    if experiment_group != "noise" else "",
                    str(res[0]) + "x" + str(res[1])
                    if experiment_group != "fov" else "", point
                    if experiment_group != "point" else "", opt_version))

            # ! Save .svg
            fig.write_image(
                "../report/{}/{}/{}/{}_{}_{}_{}_{}_{}_{}_{}_{}.svg".format(
                    dataset, scene, str(idx_frame), scene[:-2], scene[-1:],
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
                "../report/{}/{}/{}/{}/{}/{}/{}/{}_{}_{}_{}_{}_{}_{}_{}.csv".
                    format(dataset, scene, str(idx_frame), "mc"
                if motion_constraint else "!mc", noise,
                           str(res[0]) + "x" + str(res[1]), point, scene[:-2],
                           scene[-1:], str(idx_frame), "mc"
                           if motion_constraint else "!mc", noise,
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
                "../report/{}/{}/{}/{}/{}/{}/{}/{}_{}_{}_{}_{}_{}_{}_{}.csv".
                    format(dataset, scene, str(idx_frame), "mc"
                if motion_constraint else "!mc", noise,
                           str(res[0]) + "x" + str(res[1]), point, scene[:-2],
                           scene[-1:], str(idx_frame), "mc"
                           if motion_constraint else "!mc", noise,
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
                "../report/{}/{}/{}/{}/{}/{}/{}/{}_{}_{}_{}_{}_{}_{}_{}.csv".
                    format(dataset, scene, str(idx_frame), "mc"
                if motion_constraint else "!mc", noise,
                           str(res[0]) + "x" + str(res[1]), point, scene[:-2],
                           scene[-1:], str(idx_frame), "mc"
                           if motion_constraint else "!mc", noise,
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

    fig = go.Figure()

    # ! Ours method
    fig.add_trace(go.Scatter(x=x, y=_ours_m[:, 0], name='ours-rot',
                             # error_y=dict(type='data', array=_ours_std[:, 0])
                             ))
    fig.add_trace(go.Scatter(x=x, y=_ours_m[:, 1], name='ours-trans',
                             # error_y=dict(type='data', array=_ours_std[:, 1])
                             ))

    # ! 8PA
    fig.add_trace(
        go.Scatter(
            x=x,
            y=_8pa_m[:, 0],
            name='8pa-rot',
            line=dict(width=2, dash='dot'),
            # error_y=dict(type='data', array=_8pa_std[:, 0])
        ))
    fig.add_trace(
        go.Scatter(
            x=x,
            y=_8pa_m[:, 1],
            name='8pa-trans',
            line=dict(width=2, dash='dot'),
            # error_y=dict(type='data', array=_8pa_std[:, 1])
        ))

    fig.update_layout(
        title="{}_{}_{}_{}_{}_{}_{}_{}_{}_{}".format(
            dataset, scene[:-2], scene[-1:], str(idx_frame), "mc"
            if motion_constraint else "!mc", experiment_group, noise
            if experiment_group != "noise" else "",
            str(res[0]) + "x" + str(res[1])
            if experiment_group != "fov" else "", point
            if experiment_group != "point" else "", opt_version),
        xaxis_type="log",
        xaxis_title=experiment_group[0].upper() + experiment_group[1:],
        yaxis_title="Error",
        font=dict(
            family="Courier New, monospace", size=14, color="RebeccaPurple"))
    # , xaxis_type="log"

    fig.update_traces(mode='lines+markers', line_shape='linear')
    # fig.update_traces(mode='lines', line_shape='spline')
    fig.update_layout(showlegend=False)

    if experiment_group == "fov":
        fig.update_xaxes(
            ticktext=[str(res[0]) + 'x' + str(res[1]) for res in ress],
            tickvals=x,
        )

    if save:
        # ! Save .png
        fig.write_image(
            "../report/{}/{}/{}/{}_{}_{}_{}_{}_{}_{}_{}_{}.png".format(
                dataset, scene, str(idx_frame), scene[:-2], scene[-1:],
                str(idx_frame), "mc"
                if motion_constraint else "!mc", experiment_group, noise
                if experiment_group != "noise" else "",
                str(res[0]) + "x" + str(res[1])
                if experiment_group != "fov" else "", point
                if experiment_group != "point" else "", opt_version),
            scale=2)

        '''
        # ! Save .html
        fig.write_html(
            "../report/{}/{}/{}/{}_{}_{}_{}_{}_{}_{}_{}_{}.html".format(
                dataset, scene, str(idx_frame), scene[:-2], scene[-1:],
                str(idx_frame), "mc"
                if motion_constraint else "!mc", experiment_group, noise
                if experiment_group != "noise" else "",
                str(res[0]) + "x" + str(res[1])
                if experiment_group != "fov" else "", point
                if experiment_group != "point" else "", opt_version))

        # ! Save .svg
        fig.write_image(
            "../report/{}/{}/{}/{}_{}_{}_{}_{}_{}_{}_{}_{}.svg".format(
                dataset, scene, str(idx_frame), scene[:-2], scene[-1:],
                str(idx_frame), "mc"
                if motion_constraint else "!mc", experiment_group, noise
                if experiment_group != "noise" else "",
                str(res[0]) + "x" + str(res[1])
                if experiment_group != "fov" else "", point
                if experiment_group != "point" else "", opt_version))
        '''

    fig.show()


if __name__ == "__main__":
    plot_errors(noise, res, point, save=True)
    if experiment_group == "fov":
        plot_sk_values(noise, res, point, save=True)
