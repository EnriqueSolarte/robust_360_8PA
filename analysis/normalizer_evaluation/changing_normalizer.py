from read_datasets.MP3D_VO import MP3D_VO
from read_datasets.KITTI import KITTI_VO
from structures.extractor.shi_tomasi_extractor import Shi_Tomasi_Extractor
from structures.tracker import LKTracker
from analysis.utilities.camera_recovering import *
from analysis.utilities.plot_and_save_utilities import *
from analysis.utilities.experimentals_cam_recovering import *
from analysis.utilities.surfaces_utilities import get_eval_of_8PA


def run_sequence(**kwargs):
    if "filename" not in kwargs.keys():
        kwargs["filename"] = get_file_name(**kwargs, file_src=__file__)
    while True:
        kwargs, ret = get_bearings(**kwargs)
        if not ret:
            break

        kwargs = get_eval_of_8PA(**kwargs)
        bearings_kf = kwargs["bearings"]["kf"]
        bearings_frm = kwargs["bearings"]["frm"]

        prior_motion = kwargs["8PA"]["cam_pose"][0:3, 3]
        rot_fwm = get_rot_from_directional_vectors(
            vector_target=(0, 0, 1),
            vector_src=prior_motion
        )
        bearings_kf_rot = rot_fwm.T @ bearings_kf
        bearings_frm_rot = rot_fwm.T @ bearings_frm

        # norm_bearings_kf, t1 = normalizer_Hartley_isotropic(bearings_kf)
        # norm_bearings_frm, t2 = normalizer_Hartley_non_isotropic(bearings_frm)
        norm_bearings_kf, t1 = normalizer_3dv2020(bearings_kf_rot, s=2, k=10)
        norm_bearings_frm, t2 = normalizer_3dv2020(bearings_frm_rot, s=2, k=10)

        e_hat_norm = g8p().compute_essential_matrix(
            x1=norm_bearings_kf,
            x2=norm_bearings_frm
        )
        e_hat = t1.T @ e_hat_norm @ t2
        kwargs["cam_norm_8pa"] = g8p().recover_pose_from_e(
            E=e_hat,
            x1=bearings_kf,
            x2=bearings_frm
        )
        kwargs["cam_8pa"] = kwargs["8PA"]["cam_pose"]
        print("**************************************")
        kwargs = eval_cam_pose_error(**kwargs)

    return kwargs


if __name__ == '__main__':
    path = "/home/kike/Documents/datasets/MP3D_VO"
    scene = "2azQ1b91cZZ/0"
    # scene = "pRbA3pwrgk9/0"
    # scene = "i5noydFURQK/0"
    # scene = "sT4fr6TAbpF/0"
    # scene = "1LXtFkjw3qL/0"
    # scene = "759xd9YjKW5/0"
    # basedir = "/home/justin/slam/openvslam_norm/python_scripts/synthetic_points_exp/data/3dv2020"

    # path = "/run/user/1001/gvfs/sftp:host=140.114.27.95,port=50002/NFS/kike/minos/vslab_MP3D_VO/512x1024"
    data = MP3D_VO(scene=scene, basedir=path)
    scene_settings = dict(
        data_scene=data,
        idx_frame=0,
        distance_threshold=0.5,
        res=(360, 180),
        loc=(0, 0),
        special_eval=True)
    features_setting = dict(
        feat_extractor=Shi_Tomasi_Extractor(maxCorners=500),
        tracker=LKTracker(),
        show_tracked_features=False,
        # sampling=
        extra="3DV2020_R.T",
    )

    ransac_parm = dict(
        min_samples=8,
        max_trials=RansacEssentialMatrix.get_number_of_iteration(
            p_success=0.99, outliers=0.5, min_constraint=8),
        residual_threshold=1e-5,
        verbose=True,
        use_ransac=True)

    kwargs = run_sequence(**scene_settings, **features_setting, **ransac_parm)
    plot_errors(**kwargs)
    plot_bar_errors(**kwargs)
    save_info(**kwargs)
