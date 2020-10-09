from read_datasets.MP3D_VO import MP3D_VO
from read_datasets.KITTI import KITTI_VO
from structures.extractor.shi_tomasi_extractor import Shi_Tomasi_Extractor
from structures.tracker import LKTracker
from analysis.utilities.camera_recovering import *
from analysis.utilities.plot_and_save_utilities import *
from analysis.utilities.experimentals_cam_recovering import *
from file_utilities import generate_fingerprint_time
from analysis.utilities.saved_frames import load_bearings


def run_evaluation(**kwargs):
    # ! In case of filename has nor been defined yet
    if "filename" not in kwargs.keys():
        kwargs["filename"] = get_file_name(**kwargs, file_src=__file__)
    print_log_files(kwargs["log_files"])
    while True:
        try:
            if kwargs.get("use_saved_bearings", False):
                kwargs, ret = load_bearings(**kwargs)
            else:
                kwargs, ret = get_bearings(**kwargs)
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

            # kwargs["cam_OURS_opt_res_Rtks"], kwargs["loss_OURS_RES_Rtks"] = get_cam_pose_by_opt_res_error_RtSK(**kwargs)
            kwargs["cam_OURS_opt_res_ks_Rt"], \
            kwargs["loss_OURS_RES_ks_Rt"], \
            kwargs["time_OURS_RES_ks_Rt"] = get_cam_pose_by_opt_res_error_SK_Rt(**kwargs)
            # ! Based on REPROJECTION
            # kwargs["cam_PnP_opt_rpj_Rt"], kwargs["loss_PnP"] = get_cam_pose_by_opt_rpj_Rt_pnp(**kwargs)
            # kwargs["cam_OURS_opt_prj_sk"], kwargs["loss_OURS_RPJ_ks"] = get_cam_pose_by_opt_rpj_SK(**kwargs)
            kwargs = eval_cam_pose_error(**kwargs)
            kwargs["special_eval"] = True
        except:
            print("evaluation skipped")
            kwargs["special_eval"] = False
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
        distance_threshold=0.5,
        res=(360, 180),
        loc=(0, 0),
    )
    initial_values = dict(
        iVal_Res_SK=(1, 1),
        iVal_Rpj_SK=(1, 1),
        iVal_Res_RtSK=(1, 1),
    )
    features_setting = dict(
        feat_extractor=Shi_Tomasi_Extractor(
            maxCorners=10000,
            qualityLevel=0.001,
            minDistance=7,
            blockSize=3
        ),
        tracker=LKTracker(),
        show_tracked_features=True,
        special_eval=True,
        sampling=200,
        timing_evaluation=True,
        # extra="test"
        extra=generate_fingerprint_time(),
    )

    log_settings = dict(
        log_files=(os.path.dirname(os.path.dirname(__file__)) +
                   "/utilities/camera_recovering.py",))

    kwargs = run_evaluation(**scene_settings, **features_setting,
                            **initial_values, **log_settings)

    plot_errors(**kwargs)
    # plot_time_results(**kwargs)
    plot_bar_errors(**kwargs)
    save_info(**kwargs)
