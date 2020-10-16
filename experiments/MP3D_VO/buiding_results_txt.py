import os
import pandas as pd
import numpy as np
from analysis.utilities.reading_results import *

if __name__ == '__main__':
    # results_dir = "/media/kike/HD/ICRA2021/MPD3D_VO_results_10.08"
    # results_dir = "/home/kike/Documents/Research/optimal8PA/experiments/TUM_VI/results"
    results_dir = os.path.join(os.path.dirname(__file__), "results")
    settings = dict(
        results_dir=results_dir,
        seq="0",
        # key="2020-10-12.17.6.18",
        key="ablation_ks_Rt_ab_",
        save=True,
    )
    kwargs = get_dict_results(**settings)
    eval_results(quantile=0.5, **kwargs)
