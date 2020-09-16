from read_datasets.MP3D_VO import MP3D_VO
from structures.tracker import LKTracker
from structures.extractor.shi_tomasi_extractor import Shi_Tomasi_Extractor
from analysis.utilities.camera_recovering import *
from analysis.utilities.plot_and_save_utilities import *


def run_sequence(**kwargs):
    kwargs["filename"] = get_file_name(**kwargs, file_src=__file__)

    kwargs["results"] = dict()
    kwargs["results"]["kf"] = []
    kwargs["results"]["opt_rpj_8PA_error_tran"] = []
    kwargs["results"]["norm_8pa_reprojection"] = []
    kwargs["results"]["opt_rpj_8PA_reprojection"] = []

    while True:
        kwargs, ret = get_bearings(**kwargs)
        if not ret:
            break
        kwargs["results"]["kf"].append(kwargs["tracker"].initial_frame.idx)

        # ! Norm 8PA Errors
        kwargs["cam_norm_8pa_res"], reprojection = get_cam_pose_by_opt_rpj_SK(**kwargs)
        kwargs["results"]["cam_norm_8pa_res_reprojection"].append(np.sum(reprojection ** 2))

        # ! Opt Rt in reprojection 8PA Errors
        kwargs["cam_PnP_opt_rpj"], reprojection = get_cam_pose_by_opt_rpj_Rt_pnp(**kwargs)
        kwargs["results"]["cam_PnP_opt_rpj_reprojection"].append(np.sum(reprojection ** 2))

        kwargs = eval_cam_pose_error(**kwargs)

    return kwargs


if __name__ == '__main__':
    path = "/home/kike/Documents/datasets/MP3D_VO"
    scene = "2azQ1b91cZZ/0"
    # scene = "1LXtFkjw3qL/0"
    # scene = "759xd9YjKW5/0"
    # path = "/run/user/1001/gvfs/sftp:host=140.114.27.95,port=50002/NFS/kike/minos/vslab_MP3D_VO/512x1024"
    data = MP3D_VO(scene=scene, basedir=path)

    scene_settings = dict(
        data_scene=data,
        idx_frame=idx_frame,
        distance_threshold=0.5,
        res=ress[3],
        # res=(180, 180),
        # res=(65.5, 46.4),
        loc=(0, 0),
        extra="test1"
    )

    features_setting = dict(
        feat_extractor=Shi_Tomasi_Extractor(),
        tracker=LKTracker(),
        show_tracked_features=False
    )

    ransac_parm = dict(min_samples=8,
                       max_trials=RansacEssentialMatrix.get_number_of_iteration(
                           p_success=0.99, outliers=0.5, min_constraint=8
                       ),
                       residual_threshold=1e-5,
                       verbose=True,
                       use_ransac=False
                       )

    kwargs = run_sequence(**scene_settings,
                          **features_setting,
                          **ransac_parm
                          )

    plot_errors(**kwargs)
    save_info(**kwargs)
