from analysis.utilities.plot_and_save_utilities import *
from pinhole import Pinhole
from read_datasets.KITTI import KITTI_VO
from structures.extractor.shi_tomasi_extractor import Shi_Tomasi_Extractor
from structures.tracker import LKTracker

error_n8p, error_8p = [], []


def eval_camera_pose(cam, tracker, cam_gt):
    from solvers.epipolar_constraint import EightPointAlgorithmGeneralGeometry as g8p
    from solvers.optimal8pa import Optimal8PA as norm_8pa

    # g8p_norm = norm_8pa(version=opt_version)
    g8p = g8p()

    # cam = Pinhole(shape=tracker.initial_frame.shape)
    matches = np.array(tracker.get_matches())

    kfrm = cam.pixel2normalized_vector(matches[0]).T
    frm = cam.pixel2normalized_vector(matches[1]).T

    print("Number of matches: {}".format(kfrm.shape[1]))
    cam_8p = g8p.recover_pose_from_matches(x1=kfrm.copy(), x2=frm.copy())
    # cam_n8p = g8p_norm.recover_pose_from_matches(x1=kfrm.copy(), x2=frm.copy())

    kwargs = dict()
    kwargs["iVal_Res_SK"] = (1, 1)
    kwargs["iVal_Rpj_SK"] = (1, 1)
    kwargs["iVal_Res_RtSK"] = (1, 1)
    kwargs["bearings"] = dict()
    kwargs["bearings"]["kf"] = Sphere.sphere_normalization(kfrm)
    kwargs["bearings"]["frm"] = Sphere.sphere_normalization(frm)

    cam_opt_res_SK, _ = get_cam_pose_by_opt_res_error_SK(**kwargs)
    cam_opt_res_Rt, _ = get_cam_pose_by_opt_res_error_Rt(**kwargs)
    cam_opt_res_SK_Rt, _ = get_cam_pose_by_opt_res_error_SK_Rt(**kwargs)

    print("8PA:             {}".format(
        evaluate_error_in_transformation(transform_gt=cam_gt,
                                         transform_est=cam_8p)))
    print("Opt_Res_SK:      {}".format(
        evaluate_error_in_transformation(transform_gt=cam_gt,
                                         transform_est=cam_opt_res_SK)))
    print("Opt_Res_Rt:      {}".format(
        evaluate_error_in_transformation(transform_gt=cam_gt,
                                         transform_est=cam_opt_res_Rt)))
    print("Opt_Res_SK_Rt:   {}".format(
        evaluate_error_in_transformation(transform_gt=cam_gt,
                                         transform_est=cam_opt_res_SK_Rt)))

    print("kf:{} - frm:{} - matches:{}".format(tracker.initial_frame.idx,
                                               tracker.tracked_frame.idx,
                                               len(tracker.tracks)))
    print(
        "====================================================================="
    )


if __name__ == "__main__":
    from config import *

    if dataset == "kitti":
        data = KITTI_VO(basedir=basedir, scene=scene)

    tracker = LKTracker()
    threshold_camera_distance = 5
    i = 0

    for idx in range(i, 200):
        frame_curr = Frame(**data.get_frame(idx, return_dict=True), idx=idx)
        if idx == i:
            tracker.set_initial_frame(
                initial_frame=frame_curr,
                extractor=Shi_Tomasi_Extractor(maxCorners=200))
            continue

        relative_pose = frame_curr.get_relative_pose(
            key_frame=tracker.initial_frame)
        camera_distance = np.linalg.norm(relative_pose[0:3, 3])

        if camera_distance > threshold_camera_distance:
            eval_camera_pose(cam=data.camera_projection,
                             tracker=tracker,
                             cam_gt=relative_pose)
            frame_prev = tracker.tracked_frame
            tracker.set_initial_frame(
                initial_frame=frame_prev,
                extractor=Shi_Tomasi_Extractor(maxCorners=200))
            relative_pose = frame_curr.get_relative_pose(
                key_frame=tracker.initial_frame)
            camera_distance = np.linalg.norm(relative_pose[0:3, 3])

        tracked_img = tracker.track(frame=frame_curr)

        cv2.namedWindow("preview")
        cv2.imshow("preview", tracked_img)
        cv2.waitKey(1)