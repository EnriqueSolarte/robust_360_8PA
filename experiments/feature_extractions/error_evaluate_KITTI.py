from analysis.utilities.plot_and_save_utilities import *
from pinhole import Pinhole
from read_datasets.KITTI import KITTI_VO
from structures.extractor.shi_tomasi_extractor import Shi_Tomasi_Extractor
from structures.tracker import LKTracker


def eval_camera_pose(cam, **kwargs):
    from solvers.epipolar_constraint import EightPointAlgorithmGeneralGeometry as g8p
    from solvers.optimal8pa import Optimal8PA as norm_8pa

    # g8p_norm = norm_8pa(version=opt_version)
    g8p = g8p()

    # cam = Pinhole(shape=tracker.initial_frame.shape)
    matches = np.array(kwargs["tracker"].get_matches())

    kfrm = cam.pixel2normalized_vector(matches[0]).T
    frm = cam.pixel2normalized_vector(matches[1]).T

    print("Number of matches: {}".format(kfrm.shape[1]))
    # cam_8p = g8p.recover_pose_from_matches(x1=kfrm.copy(), x2=frm.copy())
    # cam_n8p = g8p_norm.recover_pose_from_matches(x1=kfrm.copy(), x2=frm.copy())

    kwargs["bearings"] = dict()
    kwargs["bearings"]["kf"] = kfrm.copy()
    kwargs["bearings"]["frm"] = frm.copy()
    kwargs["results"]["kf"].append(kwargs["tracker"].initial_frame.idx)
    kwargs["cam_8pa"], kwargs["loss_8pa"] = get_cam_pose_by_8pa(**kwargs)

    kwargs["bearings"]["kf"] = Sphere.sphere_normalization(kfrm)
    kwargs["bearings"]["frm"] = Sphere.sphere_normalization(frm)

    # cam_opt_res_SK, _ = get_cam_pose_by_opt_res_error_SK(**kwargs)
    # cam_opt_res_Rt, _ = get_cam_pose_by_opt_res_error_Rt(**kwargs)
    # cam_opt_res_SK_Rt, _ = get_cam_pose_by_opt_res_error_SK_Rt(**kwargs)

    kwargs["cam_OURS_opt_res_ks"], kwargs[
        "loss_OURS_RES_ks"] = get_cam_pose_by_opt_res_error_SK(**kwargs)
    kwargs["cam_8pa_opt_res_Rt"], kwargs[
        "loss_RES_Rt"] = get_cam_pose_by_opt_res_error_Rt(**kwargs)
    kwargs["cam_OURS_opt_res_ks_Rt"], kwargs[
        "loss_OURS_RES_ks_Rt"] = get_cam_pose_by_opt_res_error_SK_Rt(**kwargs)

    kwargs = eval_cam_pose_error(**kwargs, _print=False)

    print("8PA:             {}".format(
        evaluate_error_in_transformation(transform_gt=kwargs["cam_gt"],
                                         transform_est=kwargs["cam_8pa"])))
    print("Opt_Res_SK:      {}".format(
        evaluate_error_in_transformation(
            transform_gt=kwargs["cam_gt"],
            transform_est=kwargs["cam_OURS_opt_res_ks"])))
    print("Opt_Res_Rt:      {}".format(
        evaluate_error_in_transformation(
            transform_gt=kwargs["cam_gt"],
            transform_est=kwargs["cam_8pa_opt_res_Rt"])))
    print("Opt_Res_SK_Rt:   {}".format(
        evaluate_error_in_transformation(
            transform_gt=kwargs["cam_gt"],
            transform_est=kwargs["cam_OURS_opt_res_ks_Rt"])))

    print("kf:{} - frm:{} - matches:{}".format(
        kwargs["tracker"].initial_frame.idx,
        kwargs["tracker"].tracked_frame.idx, len(kwargs["tracker"].tracks)))
    print(
        "====================================================================="
    )

    return kwargs


if __name__ == "__main__":
    from config import *

    if dataset == "kitti":
        data = KITTI_VO(basedir=basedir, scene=scene)

    tracker = LKTracker()
    threshold_camera_distance = 5
    i = 0

    scene_settings = dict(
        data_scene=data,
        idx_frame=0,
        distance_threshold=5,
        res=(360, 180),
        pinhole_model=True,
        # res=(180, 180),
        # res=(65.5, 46.4),
        loc=(0, 0),
        extra="Initial eval",
        special_eval=False)
    initial_values = dict(
        iVal_Res_SK=(1, 1),
        iVal_Rpj_SK=(1, 1),
        # iVal_Rpj_SK=(0.5, 0.5),
        iVal_Res_RtSK=(1, 1),
    )
    features_setting = dict(
        feat_extractor=Shi_Tomasi_Extractor(maxCorners=200),
        tracker=LKTracker(),
        show_tracked_features=True)

    ransac_parm = dict(
        min_samples=8,
        max_trials=RansacEssentialMatrix.get_number_of_iteration(
            p_success=0.99, outliers=0.5, min_constraint=8),
        residual_threshold=1e-5,
        verbose=True,
        use_ransac=False)

    log_settings = dict(log_files=(os.path.dirname(os.path.dirname(__file__)) +
                                   "/utilities/camera_recovering.py", ))

    # kwargs = dict()
    kwargs = scene_settings
    kwargs.update(initial_values)
    kwargs.update(features_setting)
    kwargs.update(ransac_parm)
    kwargs.update(log_settings)
    kwargs["filename"] = get_file_name(**kwargs, file_src=__file__)
    kwargs["results"] = dict()
    kwargs["results"]["kf"] = []

    # data.number_frames
    for idx in range(i, 200):
        frame_curr = Frame(**data.get_frame(idx, return_dict=True), idx=idx)
        if idx == i:
            kwargs["tracker"].set_initial_frame(
                initial_frame=frame_curr,
                extractor=Shi_Tomasi_Extractor(maxCorners=200))
            continue

        relative_pose = frame_curr.get_relative_pose(
            key_frame=kwargs["tracker"].initial_frame)
        camera_distance = np.linalg.norm(relative_pose[0:3, 3])

        if camera_distance > threshold_camera_distance:
            kwargs["cam_gt"] = relative_pose
            kwargs = eval_camera_pose(cam=data.camera_projection, **kwargs)
            frame_prev = kwargs["tracker"].tracked_frame
            kwargs["tracker"].set_initial_frame(
                initial_frame=frame_prev,
                extractor=Shi_Tomasi_Extractor(maxCorners=200))
            relative_pose = frame_curr.get_relative_pose(
                key_frame=kwargs["tracker"].initial_frame)
            camera_distance = np.linalg.norm(relative_pose[0:3, 3])

        tracked_img = kwargs["tracker"].track(frame=frame_curr)

        cv2.namedWindow("preview")
        cv2.imshow("preview", tracked_img)
        cv2.waitKey(1)

    plot_errors(**kwargs)
    plot_bar_errors(**kwargs)
    save_info(**kwargs)
