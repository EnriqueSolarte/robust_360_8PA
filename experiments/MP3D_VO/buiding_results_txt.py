import os
import pandas as pd
import numpy as np
from analysis.utilities.reading_results import *

if __name__ == '__main__':
    # results_dir = "/media/kike/HD/ICRA2021/MPD3D_VO_results_10.08"
    # results_dir = "/home/kike/Documents/Research/optimal8PA/experiments/TUM_VI/results"
    results_dir = os.path.join(os.path.dirname(__file__), "results")
    # ! L2**2
    # extra = "ABLATION_KS:L1_RT:L2_RTKS:a-L2_KS-RT:a-L2_TRK-FEATURES"
    # extra = "ABLATION_KS:L1_RT:L2_RTKS:a-L2_KS-RT:a-L2_INLIERS_0.5"
    # ! L1**2
    # extra = "ABLATION_KS:L1_RT:L1_RTKS:a-L1_KS-RT:a-L1_TRK-FEATURES"
    # extra = "ABLATION_KS:L1_RT:L1_RTKS:a-L1_KS-RT:a-L1_INLIERS_0.5"

    # extra = "ABLATION_KS:L1_RT:L1_RTKS:b-L1_KS-RT:b-L1_TRK-FEATURES"
    # extra = "ABLATION_KS:L1_RT:L1_RTKS:b-L1_KS-RT:b-L1_INLIERS_0.5"
    # ! L1
    # extra = "ABLATION_KS:*L1_RT:*L1_RTKS:a-*L1_KS-RT:a-*L1_TRK-FEATURES"
    # extra = "ABLATION_KS:*L1_RT:*L1_RTKS:a-*L1_KS-RT:a-*L1_INLIERS_0.5"

    extra = "2020-10-17.16.32.31"
    settings = dict(
        results_dir=results_dir,
        seq="0",
        # key="2020-10-12.17.6.18",
        key=extra,
        save=True,
    )
    kwargs = get_dict_results(**settings)
    eval_results(quantile=0.5, **kwargs)
