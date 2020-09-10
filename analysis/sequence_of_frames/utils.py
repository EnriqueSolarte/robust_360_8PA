import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from utilities.data_utilities import *


def get_file_name(**kwargs):
    scene_ = os.path.dirname(kwargs["data_scene"].scene)
    filename = "Seq_frames_" + scene_
    filename += "_res_" + str(kwargs["res"][0]) + "." + str(kwargs["res"][1])
    filename += "_dist" + str(kwargs["distance_threshold"])
    if kwargs["use_ransac"]:
        filename += "_RANSAC_thr" + str(kwargs["residual_threshold"])
        filename += "_trials" + str(kwargs["max_trials"])
    filename += "_" + kwargs["extra"]

    return filename


def plot(**kwargs):
    results = list(kwargs["results"].keys())
    titles = [dt for dt in results if dt not in ("kf",)]

    fig = make_subplots(subplot_titles=["Rotation Error", "Trans Error"],
                        rows=3,
                        cols=1,
                        specs=[[{}], [{}], [{}]])

    idxs = np.linspace(0, 3, 4).reshape(-1, 1)

    dt_results = list()
    dt_results.append([dt for dt in titles if "rot" in dt])
    dt_results.append([dt for dt in titles if "tran" in dt])
    dt_results.append([dt for dt in titles if "residuals" in dt or "reprojection" in dt])

    for i, dt in enumerate(dt_results):
        if len(dt) == 0:
            continue
        loc = np.squeeze(np.where(idxs == i))
        row, col = loc[0] + 1, loc[1] + 1
        y_label = dt[0]
        for dt_r in dt:
            if "norm" in dt_r:
                color = COLOR_NORM_8PA
            elif "8pa" in dt_r:
                color = COLOR_8PA
            elif "opt_rpj" in dt_r:
                color = COLOR_OPT_RPJ_RT
            elif "opt_res" in dt_r:
                color = COLOR_OPT_RES_RT

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
    fig.write_html("plots/{}".format(fig_file))
