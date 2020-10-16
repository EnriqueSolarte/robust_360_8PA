from read_datasets.MP3D_VO import MP3D_VO
from solvers.ransac.ransac_8pa import RANSAC_8PA
from analysis.utilities.essential_e_recovering import *
from structures.tracker import LKTracker
from structures.extractor.shi_tomasi_extractor import Shi_Tomasi_Extractor
from analysis.utilities.plot_utilities import *
from analysis.utilities.save_and_load_data import save_data


def eval_ransac_8pa(**kwargs):
    if "filename" not in kwargs.keys():
        kwargs["filename"] = get_file_name(**kwargs, file_src=__file__)

    ransac = RANSAC_8PA(**kwargs)
    ransac.post_function_evaluation = kwargs.get("post_function_evaluation", get_e_by_8pa)

    if "post_function_evaluation" in kwargs.keys():
        if "8pa" in str(kwargs.get("post_function_evaluation", get_e_by_8pa)):
            kwargs["method"] = "8PA"
        if "Rt" in str(kwargs["post_function_evaluation"]):
            if "SK" in str(kwargs["post_function_evaluation"]):
                kwargs["method"] = "KS_Rt"
            else:
                kwargs["method"] = "Rt"
    while True:
        kwargs, ret = get_bearings(**kwargs)
        if not ret:
            break
        kwargs["cam_{}".format(kwargs["method"])] = ransac.get_cam_pose(
            bearings_1=kwargs["bearings"]["kf"].copy(),
            bearings_2=kwargs["bearings"]["frm"].copy(),
        )

        kwargs["time_{}".format(kwargs["method"])] = ransac.time_evaluation
        kwargs["loss_{}".format(kwargs["method"])] = ransac.best_evaluation
        kwargs["inliers_{}".format(kwargs["method"])] = ransac.best_inliers_num
        kwargs["iterations_{}".format(kwargs["method"])] = ransac.counter_trials

        kwargs = eval_cam_pose_error(**kwargs)
        print("time_: {}".format(ransac.time_evaluation))
        print("Inliers: {}".format(ransac.best_inliers_num))
        print("Iterations: {}".format(ransac.counter_trials))
        if kwargs.get("save_results", False):
            kwargs["saved_list"] = [
                "time_{}".format(kwargs["method"]),
                "loss_{}".format(kwargs["method"]),
                "inliers_{}".format(kwargs["method"]),
                "iterations_{}".format(kwargs["method"])
            ]
            save_data(**kwargs)

    return kwargs


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
        sampling=200,
    )
    features_setting = dict(
        feat_extractor=Shi_Tomasi_Extractor(maxCorners=1000),
        tracker=LKTracker(),
        show_tracked_features=False,
        special_eval=True,
        distance_threshold=0.5
    )

    ransac_settings = dict(
        expected_inliers=0.5,
        probability_success=0.99,
        residual_threshold=1e-5,
        post_function_evaluation=get_e_by_8pa,
        timing_evaluation=True,
        save_results=True
    )

    kwargs = eval_ransac_8pa(
        **scene_settings,
        **ransac_settings,
        **features_setting)
    # plot_errors(**kwargs)
    # plot_time_results(**kwargs)
    # plot_bar_errors(**kwargs)
    # save_info(**kwargs)
