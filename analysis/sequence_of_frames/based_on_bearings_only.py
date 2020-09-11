from read_datasets.MP3D_VO import MP3D_VO
from structures.tracker import LKTracker
from structures.extractor.shi_tomasi_extractor import Shi_Tomasi_Extractor
from analysis.sequence_of_frames.camera_recovering import *
from analysis.sequence_of_frames.utils import *


def run_sequence(**kwargs):
    kwargs["results"] = dict()
    kwargs["results"]["kf"] = []
    kwargs["results"]["8pa_error_rot"] = []
    kwargs["results"]["8pa_error_tran"] = []
    kwargs["results"]["norm_8pa_error_rot"] = []
    kwargs["results"]["norm_8pa_error_tran"] = []
    kwargs["results"]["opt_res_error_rot"] = []
    kwargs["results"]["opt_res_error_tran"] = []

    kwargs["results"]["8pa_residuals"] = []
    kwargs["results"]["norm_8pa_residuals"] = []
    kwargs["results"]["opt_res_residuals"] = []

    while True:
        kwargs, ret = get_bearings(**kwargs)
        if not ret:
            break
        kwargs["results"]["kf"].append(kwargs["tracker"].initial_frame.idx)

        # ! 8PA Errors
        cam_hat, residuals = get_cam_pose_by_8pa(**kwargs)
        error = evaluate_error_in_transformation(
            transform_gt=kwargs["cam_gt"],
            transform_est=cam_hat)
        kwargs["results"]["8pa_error_rot"].append(error[0])
        kwargs["results"]["8pa_error_tran"].append(error[1])
        kwargs["results"]["8pa_residuals"].append(np.sum(residuals ** 2))

        # ! Norm 8PA Errors
        cam_hat, residuals = get_cam_pose_by_opt_res_norm_8pa(**kwargs)
        error = evaluate_error_in_transformation(
            transform_gt=kwargs["cam_gt"],
            transform_est=cam_hat)
        kwargs["results"]["norm_8pa_error_rot"].append(error[0])
        kwargs["results"]["norm_8pa_error_tran"].append(error[1])
        kwargs["results"]["norm_8pa_residuals"].append(np.sum(residuals ** 2))

        # ! Opt Rt residuals 8PA Errors
        cam_hat, residuals = get_cam_pose_by_opt_res_rt(**kwargs)
        error = evaluate_error_in_transformation(
            transform_gt=kwargs["cam_gt"],
            transform_est=cam_hat)
        kwargs["results"]["opt_res_error_rot"].append(error[0])
        kwargs["results"]["opt_res_error_tran"].append(error[1])
        kwargs["results"]["opt_res_residuals"].append(np.sum(residuals ** 2))

        print("----------------------------------------------------------------------------")
        print("8pa Error-rot:           {}".format(np.median(kwargs["results"]["8pa_error_rot"], axis=0)))
        print("norm 8pa Error-rot:      {}".format(np.median(kwargs["results"]["norm_8pa_error_rot"], axis=0)))
        print("Opt Rt res Error-tran:   {}".format(np.median(kwargs["results"]["opt_res_error_rot"], axis=0)))

        print("8pa Error-tran:          {}".format(np.median(kwargs["results"]["8pa_error_tran"], axis=0)))
        print("norm 8pa Error-tran:     {}".format(np.median(kwargs["results"]["norm_8pa_error_tran"], axis=0)))
        print("Opt Rt res Error-tran:   {}".format(np.median(kwargs["results"]["opt_res_error_tran"], axis=0)))

    kwargs["filename"] = "error_8PA_seq_frames_{}".format(get_file_name(**kwargs))
    plot(**kwargs)
    save_results(**kwargs)


if __name__ == '__main__':
    path = "/home/kike/Documents/datasets/MP3D_VO"
    scene = "2azQ1b91cZZ/0"
    # scene = "1LXtFkjw3qL/0"
    # scene = "759xd9YjKW5/0"
    # path = "/run/user/1001/gvfs/sftp:host=140.114.27.95,port=50002/NFS/kike/minos/vslab_MP3D_VO/512x1024"
    data = MP3D_VO(scene=scene, basedir=path)

    scene_settings = dict(
        data_scene=data,
        idx_frame=0,
        distance_threshold=0.5,
        res=(360, 180),
        # res=(180, 180),
        # res=(65.5, 46.4),
        loc=(0, 0),
        extra="only_bearings",
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
                       use_ransac=True
                       )

    run_sequence(**scene_settings,
                 **features_setting,
                 **ransac_parm
                 )
