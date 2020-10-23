import plotly.graph_objects as go
from plotly.subplots import make_subplots
from analysis.utilities.data_utilities import *
import os
import numpy as np
from file_utilities import create_dir, save_obj
import shutil


def get_file_name(**kwargs):
    dirname, name_src = os.path.split(kwargs["file_src"])
    dirname = os.path.join(dirname, "results")
    dirname = os.path.join(dirname, kwargs["data_scene"].scene)
    name = os.path.splitext(name_src)[0]
    scene_ = os.path.dirname(kwargs["data_scene"].scene)
    filename = name + "_" + scene_
    try:
        filename += "_ifrm_{}".format(kwargs["idx_frame"])
        filename += "_res" + str(kwargs["res"][0]) + "." + str(kwargs["res"][1])
        filename += "_loc" + str(kwargs["loc"][0]) + "." + str(kwargs["loc"][1])
        if "distance_threshold" in kwargs.keys():
            filename += "_dist" + str(kwargs["distance_threshold"])
        if "noise" in kwargs.keys():
            filename += "_Noise." + str(kwargs["noise"]) + "_In." + str(kwargs["inliers_ratio"])
    except:
        print("Some labels cannot be read")
    try:
        filename += "_RANSAC_in" + str(kwargs["expected_inliers"])
        filename += "_rthr8pa" + str(kwargs.get("residual_threshold_8PA", ""))
        filename += "_set" + str(kwargs["min_super_set"])
        filename += "_thr" + str(kwargs["residual_threshold"])
        filename += "_rthr" + str(kwargs["relaxed_threshold"])
    except:
        print("RANSAC parameters not available")

    if "post_function_evaluation" in kwargs.keys():
        if "8pa" in str(kwargs["post_function_evaluation"]):
            filename += "_method_8PA"
            kwargs["method"] = "8PA"
        if "Rt" in str(kwargs["post_function_evaluation"]):
            if "SK" in str(kwargs["post_function_evaluation"]):
                filename += "_method_KS_Rt"
                kwargs["method"] = "KS_Rt"
            else:
                filename += "_method_Rt"
                kwargs["method"] = "Rt"
    if 'extra' in kwargs.keys():
        filename += "_" + kwargs["extra"]

    if 'sampling' in kwargs.keys():
        filename += "_samples" + str(kwargs["sampling"])

    try:
        initial_val = [dt for dt in kwargs.keys() if "iVal" in dt]
        if len(initial_val) > 0:
            for val in initial_val:
                filename += "_" + val + "." + str(kwargs[val])
    except:
        print("initail values not available")

    dirname = os.path.join(dirname, filename)
    create_dir(dirname, delete_previous=False)

    return os.path.join(dirname, filename)


def plot_bar_errors_and_time(**kwargs):
    results = list(kwargs["results"].keys())
    titles = [dt for dt in results if dt not in ("kf",)]

    dt_results = list()
    dt_results.append([dt for dt in titles if "rot" in dt])
    dt_results.append([dt for dt in titles if "tran" in dt])
    dt_results.append([dt for dt in titles if "time" in dt])
    n = 3 * len(dt_results)
    idxs = np.linspace(0, n - 1, n).reshape(3, -1)
    fig = make_subplots(
        subplot_titles=["Q75", "Q50", "Q25", "Q75", "Q50", "Q25", "Q75", "Q50", "Q25"],
        rows=idxs.shape[0],
        cols=idxs.shape[1],
        specs=[[{}, {}, {}], [{}, {}, {}], [{}, {}, {}]])

    kwargs["quartiles"] = dict()
    for i_row, dt in enumerate(dt_results):

        if len(dt) == 0:
            continue

        quartiles_settings = dict(Q75=(np.quantile, 0.75),
                                  Q50=(np.quantile, 0.50),
                                  Q25=(np.quantile, 0.25))

        for i_quart, quartile in enumerate(list(quartiles_settings.keys())):
            row, col = i_row + 1, i_quart + 1
            y_label = dt[0]
            for dt_r in dt:
                func = quartiles_settings[quartile][0]
                arg = quartiles_settings[quartile][1]
                kwargs["quartiles"][dt_r + "_" + quartile] = func(
                    kwargs["results"][dt_r], arg)

                color = get_color(dt_r)

                fig.add_trace(go.Bar(x=(dt_r,),
                                     y=(kwargs["quartiles"][dt_r + "_" +
                                                            quartile],),
                                     name=dt_r + "_" + quartile,
                                     marker_color=color),
                              row=row,
                              col=col)
                if "rot" in dt_r:
                    y_label = "Rotation Error"
                elif "tran" in dt_r:
                    y_label = "Translation Error"
                elif "time" in dt_r:
                    y_label = "Computation Time"

            fig.update_yaxes(title_text=y_label, row=row, col=col)

    fig_file = "{}_plot_bar_errors.html".format(kwargs["filename"])
    fig.update_layout(title_text=fig_file, height=800, width=1800)
    fig.show()
    fig.write_html("{}".format(fig_file))
    print("done")


def plot_bar_errors(**kwargs):
    results = list(kwargs["results"].keys())
    titles = [dt for dt in results if dt not in ("kf",)]

    dt_results = list()
    dt_results.append([dt for dt in titles if "rot" in dt])
    dt_results.append([dt for dt in titles if "tran" in dt])

    n = 6
    idxs = np.linspace(0, n - 1, n).reshape(2, -1)
    fig = make_subplots(
        subplot_titles=["Q75", "Q50", "Q25", "Q75", "Q50", "Q25"],
        rows=idxs.shape[0],
        cols=idxs.shape[1],
        specs=[[{}, {}, {}], [{}, {}, {}]])

    kwargs["quartiles"] = dict()
    for i_row, dt in enumerate(dt_results):

        if len(dt) == 0:
            continue

        quartiles_settings = dict(Q75=(np.quantile, 0.75),
                                  Q50=(np.quantile, 0.50),
                                  Q25=(np.quantile, 0.25))

        for i_quart, quartile in enumerate(list(quartiles_settings.keys())):
            row, col = i_row + 1, i_quart + 1
            y_label = dt[0]
            for dt_r in dt:
                func = quartiles_settings[quartile][0]
                arg = quartiles_settings[quartile][1]
                kwargs["quartiles"][dt_r + "_" + quartile] = func(
                    kwargs["results"][dt_r], arg)

                color = get_color(dt_r)

                fig.add_trace(go.Bar(x=(dt_r,),
                                     y=(kwargs["quartiles"][dt_r + "_" +
                                                            quartile],),
                                     name=dt_r + "_" + quartile,
                                     marker_color=color),
                              row=row,
                              col=col)
                if "rot" in dt_r:
                    y_label = "Rotation Error"
                elif "tran" in dt_r:
                    y_label = "Translation Error"

            fig.update_yaxes(title_text=y_label, row=row, col=col)

    fig_file = "{}_plot_bar_errors.html".format(kwargs["filename"])
    fig.update_layout(title_text=fig_file, height=800, width=1800)
    fig.show()
    fig.write_html("{}".format(fig_file))
    print("done")


def plot_errors(**kwargs):
    results = list(kwargs["results"].keys())
    titles = [dt for dt in results if dt not in ("kf",)]

    dt_results = list()
    dt_results.append([dt for dt in titles if "rot" in dt])
    dt_results.append([dt for dt in titles if "tran" in dt])
    res = [dt for dt in titles if "loss" in dt]
    if len(res) > 1:
        dt_results.append([dt for dt in titles if "loss" in dt])

    n = len(dt_results)
    idxs = np.linspace(0, n, n + 1).reshape(-1, 1)
    specs = []
    for _ in dt_results:
        specs.append([{}])
    fig = make_subplots(subplot_titles=["Rotation Error", "Trans Error"],
                        rows=n,
                        cols=1,
                        specs=specs)

    for i, dt in enumerate(dt_results):
        if len(dt) == 0:
            continue
        loc = np.squeeze(np.where(idxs == i))
        row, col = loc[0] + 1, loc[1] + 1
        y_label = dt[0]
        for dt_r in dt:
            color = get_color(dt_r)

            fig.add_trace(go.Scatter(x=kwargs["results"]["kf"],
                                     y=kwargs["results"][dt_r],
                                     name=dt_r,
                                     line=dict(color=color)),
                          row=row,
                          col=col)
            if "rot" in dt_r:
                y_label = "Rotation Error"
            elif "tran" in dt_r:
                y_label = "Translation Error"

        fig.update_xaxes(title_text="Kfrm idx", row=row, col=col)
        fig.update_yaxes(title_text=y_label, row=row, col=col)

    fig_file = "{}_plot_errors.html".format(kwargs["filename"])
    fig.update_layout(title_text=fig_file, height=800, width=1800)
    fig.show()
    fig.write_html("{}".format(fig_file))


def plot_time_results(**kwargs):
    results = list(kwargs["results"].keys())
    titles = [dt for dt in results if dt not in ("kf",)]

    dt_results = list()
    dt_results.append([dt for dt in titles if "time" in dt])
    n = 2  # !  two rows. (1) time per Kf and (2) bars 75% 50% 25%
    kwargs["quartiles"] = dict()

    fig = make_subplots(subplot_titles=["Timing per frame", "Q75", "Q50", "Q25"],
                        rows=n,
                        cols=3,
                        specs=[[{"colspan": 3}, None, None],
                               [{}, {}, {}]]
                        )

    for i, dt in enumerate(dt_results):
        if len(dt) == 0:
            continue
        row, col = 1, 1
        y_label = "Time (s)"
        for dt_r in dt:
            color = get_color(dt_r)
            fig.add_trace(go.Scatter(x=kwargs["results"]["kf"],
                                     y=kwargs["results"][dt_r],
                                     name=dt_r,
                                     line=dict(color=color)),
                          row=row,
                          col=col)

        quartiles_settings = dict(Q75=(np.quantile, 0.75),
                                  Q50=(np.quantile, 0.50),
                                  Q25=(np.quantile, 0.25))

        for i_quart, quartile in enumerate(list(quartiles_settings.keys())):
            row, col = 2, i_quart + 1
            for dt_r in dt:
                func = quartiles_settings[quartile][0]
                arg = quartiles_settings[quartile][1]
                kwargs["quartiles"][dt_r + "_" + quartile] = func(
                    kwargs["results"][dt_r], arg)

                color = get_color(dt_r)
                fig.add_trace(go.Bar(x=(dt_r,),
                                     y=(kwargs["quartiles"][dt_r + "_" +
                                                            quartile],),
                                     name=dt_r + "_" + quartile,
                                     marker_color=color),
                              row=row,
                              col=col)
            fig.update_yaxes(title_text=y_label, row=row, col=col)

        fig.update_xaxes(title_text="Kfrm idx", row=row, col=col)
        fig.update_yaxes(title_text=y_label, row=row, col=col)

    fig_file = "{}_time_results.html".format(kwargs["filename"])
    fig.update_layout(title_text=fig_file, height=800, width=1800)
    fig.show()
    fig.write_html("{}".format(fig_file))


def get_color(label):
    if "OURS" in label:
        return colors["COLOR_OURS_NORM_8PA"]
    else:
        return colors["COLOR_8PA"]


# if "OURS" in label:
#     if "opt" in label:
#         return colors["COLOR_OURS_OPT_RES_RT"]
#     return colors["COLOR_OURS_NORM_8PA"]
# elif "8pa" in label:
#     if "hartley" in label:
#         return colors["COLOR_HARTLEY_8PA"]
#     if "norm" in label:
#         return colors["COLOR_NORM"]
#     return colors["COLOR_8PA"]
# else:
#     return colors["COLOR_GENERAL"]


def save_info(only_results=True, **kwargs):
    filename = kwargs["filename"]
    if only_results:
        dir_output = os.path.join("{}.results".format(filename))
        save_obj(dir_output, kwargs["results"])
    else:
        dir_output = os.path.join("{}.kwargs".format(filename))
        save_obj(dir_output, kwargs)
    try:
        dir_output = os.path.join("{}_log.txt".format(filename))
        shutil.move(src=os.path.join(os.environ['HOME'], "log.txt"),
                    dst=dir_output)
    except:
        print("Log file was not saved")


def save_surface_results(**kwargs):
    filename = kwargs["filename"]
    dir_output = os.path.join("plots/{}.data".format(filename))

    dt = dict(results=kwargs["results"],
              v_grid=kwargs["v_grid"],
              vv_grid=kwargs["vv_grid"])
    save_obj(dir_output, dt)


def save_surfaces(**kwargs):
    filename = kwargs["filename"]
    dir_output = os.path.join("{}.data".format(filename))
    save_obj(dir_output, kwargs["surfaces"])


def print_log_files(list_files):
    for file in list_files:
        try:
            with open(file, 'r') as f:
                print(f.read())
        except:
            print("Could not register log file{}".format(file))
