import os
import pandas as pd
import numpy as np
from analysis.utilities.reading_results import *

if __name__ == '__main__':
    # results_dir = "/media/kike/HD/ICRA2021/MPD3D_VO_results_10.08"
    results_dir = os.path.join(os.path.dirname(__file__), "results")
    settings = dict(
        results_dir=results_dir,
        seq="0",
        key="ablation_ks_Rt_a_",
        # key="2020-10-12.12.23.12",
        # key="all_scenes_evaluations",
        save=False,
        ext=".results"
    )
    kwargs = get_results(**settings)
    eval_results(quantile=0.5, **kwargs)
