from read_datasets.MP3D_VO import MP3D_VO
from structures.tracker import LKTracker
from structures.extractor.shi_tomasi_extractor import Shi_Tomasi_Extractor
from solvers.epipolar_constraint_by_ransac import RansacEssentialMatrix

from analysis.sequence_in_360_fov.based_on_bearings_only import plot_errors, plot_bar_errors, run_sequence, save_results

import numpy as np

if __name__ == '__main__':
    from config import *

    data = MP3D_VO(scene=scene, basedir=basedir)

    scene_settings = dict(
        data_scene=data,
        idx_frame=idx_frame,
        distance_threshold=0.5,
        res=ress[3],
        loc=(0, 0),
        extra="test1")

    features_setting = dict(
        feat_extractor=Shi_Tomasi_Extractor(),
        tracker=LKTracker(),
        show_tracked_features=False)

    ransac_parm = dict(
        min_samples=8,
        max_trials=RansacEssentialMatrix.get_number_of_iteration(
            p_success=0.99, outliers=0.5, min_constraint=8),
        residual_threshold=1e-5,
        verbose=True,
        use_ransac=True)

    tmps = []
    n = 1
    for i in range(n):
        print("Iteration: " + str(i))
        scene_settings["idx_frame"] = idx_frame
        tmp = run_sequence(**scene_settings, **features_setting, **ransac_parm)
        save_results(**tmp)
        tmps.append(tmp)

    mean_error = dict()
    mean_error["results"] = dict()
    mean_error["results"]["kf"] = list(
        range(len(tmps[0]["results"]["8pa_error_rot"])))
    mean_error["results"]["8pa_error_rot"] = []
    mean_error["results"]["8pa_error_tran"] = []
    mean_error["results"]["norm_8pa_error_rot"] = []
    mean_error["results"]["norm_8pa_error_tran"] = []
    mean_error["results"]["opt_res_error_rot"] = []
    mean_error["results"]["opt_res_error_tran"] = []
    mean_error["filename"] = tmps[0]["filename"]

    for i in range(len(tmps[0]["results"]["8pa_error_rot"])):
        mean_error["results"]["8pa_error_rot"].append(
            np.mean(
                [tmps[j]["results"]["8pa_error_rot"][i] for j in range(n)]))
        mean_error["results"]["norm_8pa_error_rot"].append(
            np.mean([
                tmps[j]["results"]["norm_8pa_error_rot"][i] for j in range(n)
            ]))
        mean_error["results"]["opt_res_error_rot"].append(
            np.mean([
                tmps[j]["results"]["opt_res_error_rot"][i] for j in range(n)
            ]))

        mean_error["results"]["8pa_error_tran"].append(
            np.mean(
                [tmps[j]["results"]["8pa_error_tran"][i] for j in range(n)]))
        mean_error["results"]["norm_8pa_error_tran"].append(
            np.mean([
                tmps[j]["results"]["norm_8pa_error_tran"][i] for j in range(n)
            ]))
        mean_error["results"]["opt_res_error_tran"].append(
            np.mean([
                tmps[j]["results"]["opt_res_error_tran"][i] for j in range(n)
            ]))

    plot_errors(**mean_error)
    plot_bar_errors(**mean_error)
    save_results(**mean_error)
