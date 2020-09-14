import plotly.graph_objects as go
from plotly.subplots import make_subplots
from analysis.utilities.data_utilities import *
from file_utilities import create_dir
from varname import nameof
import shutil


def get_file_name(**kwargs):
    dirname, name_src = os.path.split(kwargs["file_src"])
    dirname = os.path.join(dirname, "plots")
    create_dir(dirname, delete_previous=False)
    name = os.path.splitext(name_src)[0]
    scene_ = os.path.dirname(kwargs["data_scene"].scene)
    filename = name + "_" + scene_
    filename += "_ifrm_{}".format(kwargs["idx_frame"])
    filename += "_res_" + str(kwargs["res"][0]) + "." + str(kwargs["res"][1])
    filename += "_dist" + str(kwargs["distance_threshold"])
    if kwargs["use_ransac"]:
        filename += "_RANSAC_thr" + str(kwargs["residual_threshold"])
        filename += "_trials" + str(kwargs["max_trials"])
    filename += "_" + kwargs["extra"]

    initial_val = [dt for dt in kwargs.keys() if "iVal" in dt]
    if len(initial_val) > 0:
        for val in initial_val:
            filename += "_" + val + "." + str(kwargs[val])

    return os.path.join(dirname, filename)


def plot_bar_errors(**kwargs):
    results = list(kwargs["results"].keys())
    titles = [dt for dt in results if dt not in ("kf",)]

    dt_results = list()
    dt_results.append([dt for dt in titles if "rot" in dt])
    dt_results.append([dt for dt in titles if "tran" in dt])

    n = 6
    idxs = np.linspace(0, n - 1, n).reshape(2, -1)
    fig = make_subplots(subplot_titles=["Q75", "Q50", "Q25", "Q75", "Q50", "Q25"],
                        rows=idxs.shape[0],
                        cols=idxs.shape[1],
                        specs=[[{}, {}, {}], [{}, {}, {}]])

    kwargs["quartiles"] = dict()
    for i_row, dt in enumerate(dt_results):

        if len(dt) == 0:
            continue

        quartiles_settings = dict(
            Q75=(np.quantile, 0.75),
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

                fig.add_trace(go.Bar(
                    x=(dt_r,),
                    y=(kwargs["quartiles"][dt_r + "_" + quartile],),
                    name=dt_r + "_" + quartile,
                    marker_color=color
                ), row=row, col=col)
                if "rot" in dt_r:
                    y_label = "Rotation Error"
                elif "tran" in dt_r:
                    y_label = "Translation Error"

            fig.update_yaxes(title_text=y_label, row=row, col=col)

    fig_file = "{}.html".format(kwargs["filename"])
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
    res = [dt for dt in titles if "residuals_error" in dt or "reprojection" in dt]
    if len(res) > 1:
        dt_results.append([dt for dt in titles if "residuals_error" in dt or "reprojection" in dt])

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
    fig.write_html("{}".format(fig_file))


def get_color(label):
    if "OURS" in label:
        return COLOR_NORM_8PA_OURS
    elif "8pa" in label:
        if "8pa_opt_res" in label:
            return COLOR_OPT_RPJ_RT_PNP
        else:
            return COLOR_8PA
    elif "PnP" in label:
        return COLOR_OPT_RES_RT


def save_info(only_results=True, **kwargs):
    filename = kwargs["filename"]
    if only_results:
        dir_output = os.path.join("{}.results".format(filename))
        save_obj(dir_output, kwargs["results"])
    else:
        dir_output = os.path.join("{}.kwargs".format(filename))
        save_obj(dir_output, kwargs)
    try:
        log_dir = os.path.dirname(os.path.dirname(filename))
        dir_output = os.path.join("{}_log.txt".format(filename))
        shutil.copyfile(
            src=os.path.join(log_dir, "log.txt"),
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
