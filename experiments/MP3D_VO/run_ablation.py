from analysis.sequence_in_360_fov.all_solvers import *
import os
from file_utilities import save_obj


def run_ablation(**kwargs):
    # ! In case of filename has nor been defined yet
    if "filename" not in kwargs.keys():
        kwargs["filename"] = get_file_name(**kwargs, file_src=__file__)
    while True:
        kwargs, ret = load_bearings(**kwargs)

        if not ret:
            break
        print(
            "================================================================="
        )
        print("{}".format(kwargs["filename"]))
        # ! Based on RESIDUALS
        kwargs["cam_8pa"], kwargs["loss_8pa"], kwargs["time_8pa"] = get_cam_pose_by_8pa(**kwargs)

        kwargs["cam_OURS_opt_res_ks"], kwargs["loss_OURS_RES_ks"], kwargs[
            "time_OURS_RES_ks"] = get_cam_pose_by_opt_res_error_SK(**kwargs)

        kwargs["cam_8pa_opt_res_Rt"], \
        kwargs["loss_RES_Rt"], \
        kwargs["time_RES_Rt"] = get_cam_pose_by_opt_res_error_Rt(**kwargs)

        kwargs["cam_OURS_opt_res_ks_Rt"], \
        kwargs["loss_OURS_RES_ks_Rt"],\
        kwargs["time_OURS_RES_ks_Rt"] = get_cam_pose_by_opt_res_error_SK_Rt(**kwargs)

        kwargs["cam_OURS_opt_res_Rtks"], \
        kwargs["loss_OURS_RES_Rtks"], \
        kwargs["time_OURS_RES_Rtks"] = get_cam_pose_by_opt_res_error_RtSK(**kwargs)

        kwargs = eval_cam_pose_error(**kwargs)
    return kwargs


if __name__ == '__main__':
    path = "/home/kike/Documents/datasets/MP3D_VO"
    scene_list = os.listdir(path)
    extra = generate_fingerprint_time() + "_ablation_all_rt_special_"
    for sc in ["pRbA3pwrgk9"]:
        scene = sc + "/0"
        data = MP3D_VO(scene=scene, basedir=path)
        scene_settings = dict(
            data_scene=data,
            extra=extra,
        )

        initial_values = dict(
            iVal_Res_SK=(1, 1),
            iVal_Res_RtSK=(1, 1),
            timing_evaluation=True)

        kwargs = run_ablation(**scene_settings,
                              **initial_values)

    plot_bar_errors(**kwargs)
    save_info(**kwargs)
