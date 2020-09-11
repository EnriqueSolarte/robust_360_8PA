from read_datasets.MP3D_VO import MP3D_VO
from structures.tracker import LKTracker
from structures.extractor.shi_tomasi_extractor import Shi_Tomasi_Extractor
from analysis.sequence_of_frames.camera_recovering import *
from analysis.sequence_of_frames.utils import *


def run_sequence(**kwargs):
    kwargs["results"] = dict()
    kwargs["results"]["kf"] = []
    kwargs["results"]["norm_8pa_error_rot"] = []
    kwargs["results"]["norm_8pa_error_tran"] = []
    kwargs["results"]["opt_rpj_8PA_error_rot"] = []
    kwargs["results"]["opt_rpj_8PA_error_tran"] = []
    kwargs["results"]["norm_8pa_reprojection"] = []
    kwargs["results"]["opt_rpj_8PA_reprojection"] = []

    while True:
        kwargs, ret = get_bearings(**kwargs)
        if not ret:
            break
        kwargs["results"]["kf"].append(kwargs["tracker"].initial_frame.idx)

        # ! Norm 8PA Errors
        cam_hat, reprojection = get_cam_pose_by_opt_rpj_norm_8pa(**kwargs)
        error = evaluate_error_in_transformation(
            transform_gt=kwargs["cam_gt"],
            transform_est=cam_hat)
        kwargs["results"]["norm_8pa_error_rot"].append(error[0])
        kwargs["results"]["norm_8pa_error_tran"].append(error[1])
        kwargs["results"]["norm_8pa_reprojection"].append(np.sum(reprojection ** 2))

        # ! Opt Rt in reprojection 8PA Errors
        cam_hat, reprojection = get_cam_pose_by_opt_rpj_8PA_rt(**kwargs)
        error = evaluate_error_in_transformation(
            transform_gt=kwargs["cam_gt"],
            transform_est=cam_hat)
        kwargs["results"]["opt_rpj_8PA_error_rot"].append(error[0])
        kwargs["results"]["opt_rpj_8PA_error_tran"].append(error[1])
        kwargs["results"]["opt_rpj_8PA_reprojection"].append(np.sum(reprojection ** 2))

        print("----------------------------------------------------------------------------")
        print("norm 8pa Error-rot:      {}".format(np.median(kwargs["results"]["norm_8pa_error_rot"], axis=0)))
        print("Opt Rt rpj Error-rot:    {}".format(np.median(kwargs["results"]["opt_rpj_8PA_error_rot"], axis=0)))

        print("norm 8pa Error-tran:     {}".format(np.median(kwargs["results"]["norm_8pa_error_tran"], axis=0)))
        print("Opt Rt rpj Error-tran:   {}".format(np.median(kwargs["results"]["opt_rpj_8PA_error_tran"], axis=0)))

    kwargs["filename"] = "error_8PA_seq_frames_{}".format(get_file_name(**kwargs))
    plot(**kwargs)
    save_results(**kwargs)


if __name__ == '__main__':
    from config import *
    data = MP3D_VO(scene=scene, basedir=basedir)

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

    run_sequence(**scene_settings,
                 **features_setting,
                 **ransac_parm
                 )
