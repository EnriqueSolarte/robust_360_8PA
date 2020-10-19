from analysis.sequence_in_360_fov.all_solvers import *
from analysis.utilities.save_and_load_data import load_bearings_from_sampled_pcl
import os


def run_ablation(**kwargs):
    # ! In case of filename has nor been defined yet
    if "filename" not in kwargs.keys():
        kwargs["filename"] = get_file_name(**kwargs, file_src=__file__)
    while True:
        if kwargs.get("use_synthetic_points", True):
            kwargs, ret = load_bearings_from_sampled_pcl(**kwargs)
        else:
            kwargs, ret = load_bearings_from_tracked_features(**kwargs)

        if not ret:
            break
        print(
            "================================================================="
        )
        print("{}".format(kwargs["filename"]))
        # ! Based on RESIDUALS
        kwargs["cam_8pa"], kwargs["loss_8pa"], kwargs["time_8pa"] = get_cam_pose_by_8pa(**kwargs)

        # kwargs["cam_OURS_opt_res_ks"], kwargs["loss_OURS_opt_res_ks"], kwargs[
        #     "time_OURS_opt_res_ks"] = get_cam_pose_by_opt_res_error_SK(**kwargs)

        kwargs["cam_8pa_opt_res_Rt"], \
        kwargs["loss_8pa_opt_res_Rt"], \
        kwargs["time_8pa_opt_res_Rt"] = get_cam_pose_by_opt_res_error_Rt(**kwargs)

        kwargs["cam_OURS_opt_res_ks_Rt"], \
        kwargs["loss_OURS_opt_res_ks_Rt"], \
        kwargs["time_OURS_opt_res_ks_Rt"] = get_cam_pose_by_opt_res_error_SK_Rt(**kwargs)
        #
        # kwargs["cam_OURS_opt_res_Rtks"], \
        # kwargs["loss_OURS_opt_res_Rtks"], \
        # kwargs["time_OURS_opt_res_Rtks"] = get_cam_pose_by_opt_res_error_RtSK(**kwargs)
        #
        kwargs = eval_cam_pose_error(**kwargs)
    return kwargs


if __name__ == '__main__':
    path = "/home/kike/Documents/datasets/MP3D_VO"
    scene_list = os.listdir(path)
    keyword = "inliers_0.95_noise_10"
    # ! L2**2
    # extra = "ABLATION_KS:L1_RT:L2_RTKS:a-L2_KS-RT:a-L2_TRK-FEATURES"
    # extra = "ABLATION_KS:L1_RT:L2_RTKS:a-L2_KS-RT:a-L2_INLIERS_0.5"
    # ! L1**2
    # extra = "ABLATION_KS:L1_RT:L1_RTKS:a-L1_KS-RT:a-L1_TRK-FEATURES"
    # extra = "ABLATION_KS:L1_RT:L1_RTKS:a-L1_KS-RT:a-L1_INLIERS_0.5"

    # extra = "ABLATION_KS:L1_RT:L1_RTKS:b-L1_KS-RT:b-L1_TRK-FEATURES"
    # extra = "ABLATION_KS:L1_RT:L1_RTKS:b-L1_KS-RT:b-L1_INLIERS_0.5"

    # ! L1
    # extra = "ABLATION_KS:*L1_RT:*L1_RTKS:a-*L1_KS-RT:a-*L1_TRK-FEATURES"s
    # extra = "ABLATION_KS:*L1_RT:*L1_RTKS:a-*L1_KS-RT:a-*L1_INLIERS_0.5"

    # extra = "COMPARISON_KS:L1_RT:L2_RTKS:B=0.5-a-L2_KS-RT:B=0.5-a-L1_{}".format(keyword)
    extra = "TEST_RT:L1_KS-RT:B=0.5-a-L1_{}".format(keyword)

    for sc in ("pRbA3pwrgk9",):
        scene = sc + "/0"

        data = MP3D_VO(scene=scene, basedir=path)
        scene_settings = dict(
            data_scene=data,
            extra=extra,
            use_synthetic_points=True,
            keyword=keyword
        )

        initial_values = dict(
            iVal_Res_SK=(1, 1),
            iVal_Res_RtSK=(1, 1),
            timing_evaluation=True)

        kwargs = run_ablation(**scene_settings,
                              **initial_values)

    plot_bar_errors_and_time(**kwargs)
    save_info(**kwargs)
