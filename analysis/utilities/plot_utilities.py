import plotly.graph_objects as go
from plotly.subplots import make_subplots
from analysis.utilities.data_utilities import *
from file_utilities import create_dir


def get_file_name(**kwargs):
    dirname, name_src = os.path.split(kwargs["file_src"])
    dirname = os.path.join(dirname, "plots")
    create_dir(dirname, delete_previous=False)
    name = os.path.splitext(name_src)[0]
    scene_ = os.path.dirname(kwargs["data_scene"].scene)
    filename = name + "_" + scene_
    filename += "_res_" + str(kwargs["res"][0]) + "." + str(kwargs["res"][1])
    filename += "_dist" + str(kwargs["distance_threshold"])
    if kwargs["use_ransac"]:
        filename += "_RANSAC_thr" + str(kwargs["residual_threshold"])
        filename += "_trials" + str(kwargs["max_trials"])
    filename += "_" + kwargs["extra"]

    return os.path.join(dirname, filename)


def plot_bar_errors(**kwargs):
    results = list(kwargs["results"].keys())
    titles = [dt for dt in results if dt not in ("kf", )]

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

                if "OURS" in dt_r:
                    color = COLOR_NORM_8PA
                elif "8pa" in dt_r:
                    color = COLOR_8PA
                elif "opt_rpj" in dt_r:
                    color = COLOR_OPT_RPJ_RT
                elif "opt_res" in dt_r:
                    color = COLOR_OPT_RES_RT

                fig.add_trace(go.Bar(x=(dt_r, ),
                                     y=(kwargs["quartiles"][dt_r + "_" +
                                                            quartile], ),
                                     name=dt_r + "_" + quartile,
                                     marker_color=color),
                              row=row,
                              col=col)
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
    titles = [dt for dt in results if dt not in ("kf", )]

    dt_results = list()
    dt_results.append([dt for dt in titles if "rot" in dt])
    dt_results.append([dt for dt in titles if "tran" in dt])
    res = [dt for dt in titles if "residuals" in dt or "reprojection" in dt]
    if len(res) > 1:
        dt_results.append(
            [dt for dt in titles if "residuals" in dt or "reprojection" in dt])

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
            if "norm" or "8pa" or "opt_rpj" or "opt_res" in dt_r:
                if "norm" in dt_r:
                    color = COLOR_NORM_8PA
                elif "8pa" in dt_r:
                    color = COLOR_8PA
                elif "opt_rpj" in dt_r:
                    color = COLOR_OPT_RPJ_RT
                elif "opt_res" in dt_r:
                    color = COLOR_OPT_RES_RT
                fig.add_trace(go.Scatter(x=kwargs["results"]["kf"],
                                         y=kwargs["results"][dt_r],
                                         name=dt_r,
                                         line=dict(color=color)),
                              row=row,
                              col=col)
            else:
                fig.add_trace(go.Scatter(x=kwargs["results"]["kf"],
                                         y=kwargs["results"][dt_r],
                                         name=dt_r),
                              row=row,
                              col=col)
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
