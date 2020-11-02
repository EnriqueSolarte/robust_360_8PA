from analysis.sequence_in_360_fov.all_solvers import *
from analysis.utilities.save_and_load_data import load_bearings_from_sampled_pcl
import os
from analysis.utilities.ablation_study import *


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
        kwargs["cam_8pa"], \
        kwargs["time_8pa"], \
        kwargs["sigma_8pa"], \
        kwargs["C_8pa"], \
        kwargs["pm_8pa"] = get_ablation_data_by_8pa(**kwargs)
        # # #
        kwargs["cam_OURS_opt_ks"], \
        kwargs["time_OURS_opt_ks"], \
        kwargs["sigma_OURS_opt_ks"], \
        kwargs["C_OURS_opt_ks"], \
        kwargs["pm_OURS_opt_ks"] = get_ablation_data_by_opt_SK(**kwargs)
        # #
        kwargs["cam_Hartley_isotropic"], \
        kwargs["time_Hartley_isotropic"], \
        kwargs["sigma_Hartley_isotropic"], \
        kwargs["C_Hartley_isotropic"], \
        kwargs["pm_Hartley_isotropic"] = get_ablation_data_by_Hartley_isotropic(**kwargs)

        kwargs["cam_Hartley_non_isotropic"], \
        kwargs["time_Hartley_non_isotropic"], \
        kwargs["sigma_Hartley_non_isotropic"], \
        kwargs["C_Hartley_non_isotropic"], \
        kwargs["pm_Hartley_non_isotropic"] = get_ablation_data_by_Hartley_non_isotropic(**kwargs)

        kwargs["cam_Hartley_non_iso_isotropic"], \
        kwargs["time_Hartley_non_iso_isotropic"], \
        kwargs["sigma_Hartley_non_iso_isotropic"], \
        kwargs["C_Hartley_non_iso_isotropic"], \
        kwargs["pm_Hartley_non_iso_isotropic"] = get_ablation_data_by_Hartley_non_isotropic_isotropic(**kwargs)

        # kwargs["cam_8pa_opt_Rt_L2"], \
        # kwargs["loss_8pa_opt_Rt_L2"], \
        # kwargs["time_8pa_opt_Rt_L2"] = get_cam_pose_by_opt_Rt_L2(**kwargs)
        # #
        # kwargs["cam_opt_wRt_L2"], \
        # kwargs["loss_opt_wRt_L2"], \
        # kwargs["time_opt_wRt_L2"] = get_cam_pose_by_opt_wRt_L2(**kwargs)
        # #
        # kwargs["cam_OURS_opt_ks_wRt_L2"], \
        # kwargs["loss_OURS_opt_ks_wRt_L2"], \
        # kwargs["time_OURS_opt_ks_wRt_L2"] = get_cam_pose_by_opt_SK_wRt_L2(**kwargs)
        #
        # # #
        # kwargs["cam_opt_const_wRt_L2"], \
        # kwargs["loss_opt_const_wRt_L2"], \
        # kwargs["time_opt_const_wRt_L2"] = get_cam_pose_by_opt_const_wRt_L2(
        #     **kwargs)
        # #
        # kwargs["cam_OURS_opt_ks_const_wRt_L2"], \
        # kwargs["loss_OURS_opt_ks_const_wRt_L2"], \
        # kwargs["time_OURS_opt_ks_const_wRt_L2"] = get_cam_pose_by_opt_SK_const_wRt_L2(
        #     **kwargs)

        kwargs = eval_cam_pose_error(**kwargs)
    return kwargs


if __name__ == '__main__':
    path = "/home/kike/Documents/datasets/MP3D_VO"
    scene_list = os.listdir(path)
    # keyword = "_samples_200_inliers_0.5_noise_500"
    keyword = "samples_200_dist"
    extra = "ablation_parallax_motion_{}".format(keyword)
    for sc in ("i5noydFURQK",):
        scene = sc + "/0"

        data = MP3D_VO(scene=scene, basedir=path)
        scene_settings = dict(
            data_scene=data,
            extra=extra,
            use_synthetic_points=False,
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
