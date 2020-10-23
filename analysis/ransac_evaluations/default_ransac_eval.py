from read_datasets.MP3D_VO import MP3D_VO
from analysis.utilities.data_utilities import get_bearings
from solvers.ransac.ransac_8pa import RANSAC_8PA
from solvers.ransac.ransac_opt_8pa import RANSAC_OPT_8PA
from geometry_utilities import evaluate_error_in_transformation
import pandas as pd
from structures.tracker import LKTracker
from structures.extractor.shi_tomasi_extractor import Shi_Tomasi_Extractor
from analysis.utilities.data_utilities import *
from analysis.utilities.essential_e_recovering import *
from analysis.utilities.save_and_load_data import save_data
from analysis.utilities.plot_utilities import *


def eval(method, ransac, **kwargs):
    kwargs["cam_{}".format(method)] = ransac.get_cam_pose(
        bearings_1=kwargs["bearings"]["kf"].copy(),
        bearings_2=kwargs["bearings"]["frm"].copy()
    )
    rot_e, tran_e = evaluate_error_in_transformation(
        transform_est=kwargs["cam_{}".format(method)],
        transform_gt=kwargs["cam_gt"]
    )
    kwargs = add_results(key="error_cam_{}_rot".format(method), data=rot_e, **kwargs)
    kwargs = add_results(key="error_cam_{}_tran".format(method), data=tran_e, **kwargs)
    kwargs = add_results(key="residual_{}".format(method), data=ransac.best_evaluation, **kwargs)
    kwargs = add_results(key="inliers_{}".format(method),
                         data=ransac.best_inliers_num / kwargs["bearings"]["frm"].shape[1], **kwargs)
    kwargs = add_results(key="iterations_{}".format(method), data=ransac.counter_trials, **kwargs)
    kwargs = add_results(key="time_{}".format(method), data=ransac.time_evaluation, **kwargs)
    print("method:{} - ok ".format(method))
    print("------------------------------------------------------")
    return kwargs


def default_ransac_8pa(**kwargs):
    ransac = RANSAC_8PA(**kwargs)
    return eval(method="8PA", ransac=ransac, **kwargs)


def ransac_8pa_with_opt_rt_L2(**kwargs):
    ransac = RANSAC_8PA(**kwargs)
    ransac.post_function_evaluation = get_e_by_opt_res_error_Rt_L2
    return eval(method="8PA_opt_Rt", ransac=ransac, **kwargs)


def ransac_8pa_with_opt_ks(**kwargs):
    ransac = RANSAC_8PA(**kwargs)
    ransac.post_function_evaluation = get_e_by_opt_res_error_SK
    return eval(method="OURS_8PA_opt_ks", ransac=ransac, **kwargs)


def ransac_8pa_with_opt_ksrt(**kwargs):
    ransac = RANSAC_8PA(**kwargs)
    ransac.post_function_evaluation = get_e_by_opt_res_error_SKRt
    return eval(method="OURS_8PA_opt_ksrt", ransac=ransac, **kwargs)


def eval_methods_ransac(**kwargs):
    if "filename" not in kwargs.keys():
        kwargs["filename"] = get_file_name(**kwargs, file_src=__file__)

    while True:
        kwargs, ret = get_bearings(**kwargs)
        if not ret:
            break

        kwargs = default_ransac_8pa(**kwargs)
        kwargs = ransac_8pa_with_opt_rt_L2(**kwargs)
        kwargs = ransac_8pa_with_opt_ks(**kwargs)
        kwargs = ransac_8pa_with_opt_ksrt(**kwargs)

        kwargs = eval_cam_pose_error(**kwargs)
        if kwargs.get("save_results", False):
            save_data(**kwargs)

        print(kwargs["filename"])
        print(kwargs["extra"])


if __name__ == '__main__':
    path = "/home/kike/Documents/datasets/MP3D_VO"
    scene = "2azQ1b91cZZ/0"
    # scene = "i5noydFURQK/0"
    # scene = "sT4fr6TAbpF/0"
    # scene = "1LXtFkjw3qL/0"
    # scene = "759xd9YjKW5/0"

    data = MP3D_VO(scene=scene, basedir=path)
    scene_settings = dict(
        data_scene=data,
        idx_frame=0,
        res=(360, 180),
        loc=(0, 0),
        sampling=500,
        extra=generate_fingerprint_time()
    )
    features_setting = dict(
        feat_extractor=Shi_Tomasi_Extractor(maxCorners=1000),
        tracker=LKTracker(),
        show_tracked_features=False,
        special_eval=True)

    ransac_settings = dict(
        expected_inliers=0.4,
        probability_success=0.99,
        residual_threshold=3e-5,
    )

    opt_setting = dict(
        save_results=True
    )

    eval_methods_ransac(
        **scene_settings,
        **ransac_settings,
        **features_setting,
        **opt_setting,
    )
