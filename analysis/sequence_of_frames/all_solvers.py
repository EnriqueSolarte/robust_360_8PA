from read_datasets.MP3D_VO import MP3D_VO
from structures.extractor.shi_tomasi_extractor import Shi_Tomasi_Extractor
from structures.tracker import LKTracker
from analysis.utilities.camera_recovering import *
from analysis.utilities.plot_and_save_utilities import *
from analysis.utilities.experimentals_cam_recovering import *


def run_sequence(**kwargs):
    kwargs["filename"] = get_file_name(**kwargs, file_src=__file__)
    print_log_files(kwargs["log_files"])
    kwargs["results"] = dict()
    kwargs["results"]["kf"] = []
    while True:
        kwargs, ret = get_bearings(**kwargs)
        if not ret:
            break
        print("=================================================================")
        print("{}".format(kwargs["filename"]))
        # ! Based on RESIDUALS
        kwargs["results"]["kf"].append(kwargs["tracker"].initial_frame.idx)
        kwargs["cam_8pa"], kwargs["loss_8pa"] = get_cam_pose_by_8pa(**kwargs)
        kwargs["cam_OURS_opt_res_ks"], kwargs["loss_OURS_RES_ks"] = get_cam_pose_by_opt_res_error_SK(**kwargs)
        kwargs["cam_8pa_opt_res_Rt"], kwargs["loss_RES_Rt"] = get_cam_pose_by_opt_res_error_Rt(**kwargs)
        # kwargs["cam_OURS_opt_res_Rtks"], kwargs["loss_OURS_RES_Rtks"] = get_cam_pose_by_opt_res_error_RtSK(**kwargs)
        kwargs["cam_OURS_opt_res_ks_Rt"], kwargs["loss_OURS_RES_ks_Rt"] = get_cam_pose_by_opt_res_error_SK_Rt(**kwargs)
        # ! Based on REPROJECTION
        # kwargs["cam_PnP_opt_rpj_Rt"], kwargs["loss_PnP"] = get_cam_pose_by_opt_rpj_Rt_pnp(**kwargs)
        # kwargs["cam_OURS_opt_prj_sk"], kwargs["loss_OURS_RPJ_ks"] = get_cam_pose_by_opt_rpj_SK(**kwargs)
        kwargs = eval_cam_pose_error(**kwargs)

    return kwargs


if __name__ == '__main__':
    path = "/home/kike/Documents/datasets/MP3D_VO"
    # scene = "2azQ1b91cZZ/0"
    # scene = "i5noydFURQK/0"
    # scene = "1LXtFkjw3qL/0"
    scene = "759xd9YjKW5/0"
    # path = "/run/user/1001/gvfs/sftp:host=140.114.27.95,port=50002/NFS/kike/minos/vslab_MP3D_VO/512x1024"
    data = MP3D_VO(scene=scene, basedir=path)
    print("RES KS: norm_residuals * sigma[-1]/ max(*)")
    scene_settings = dict(
        data_scene=data,
        idx_frame=0,
        distance_threshold=0.5,
        res=(360, 180),
        # res=(180, 180),
        # res=(65.5, 46.4),
        loc=(0, 0),
        extra="Initial eval",
        special_eval=False
    )
    initial_values = dict(
        iVal_Res_SK=(1, 1),
        iVal_Rpj_SK=(1, 1),
        # iVal_Rpj_SK=(0.5, 0.5),
        iVal_Res_RtSK=(1, 1),
    )
    features_setting = dict(
        feat_extractor=Shi_Tomasi_Extractor(maxCorners=200),
        tracker=LKTracker(),
        show_tracked_features=False
    )

    ransac_parm = dict(min_samples=8,
                       max_trials=RansacEssentialMatrix.get_number_of_iteration(
                           p_success=0.99, outliers=0.5, min_constraint=8
                       ),
                       residual_threshold=1e-5,
                       verbose=True,
                       use_ransac=True
                       )

    log_settings = dict(
        log_files=(
            "/home/kike/Documents/Research/optimal8PA/analysis/utilities/camera_recovering.py",
        )
    )

    kwargs = run_sequence(**scene_settings,
                          **features_setting,
                          **ransac_parm,
                          **initial_values,
                          **log_settings
                          )

    plot_errors(**kwargs)
    plot_bar_errors(**kwargs)
    save_info(**kwargs)
