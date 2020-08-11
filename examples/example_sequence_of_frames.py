from solvers.epipolar_constraint import EightPointAlgorithmGeneralGeometry as g8p
from solvers.optimal8pa import Optimal8PA as g8p_norm
from read_datasets.MP3D_VO import MP3D_VO
import cv2
from structures.extractor import ORBExtractor
from structures.tracker import LKTracker
from structures.frame import Frame
from geometry_utilities import *

global error_n8p
global error_8p


def eval_camera_pose(tracker, cam_gt):
    cam = Sphere(shape=tracker.initial_frame.shape)
    matches = tracker.get_matches()
    # rot = get_rot_from_directional_vectors((0, 0, 1), (0, 0, 1))
    bearings_kf = cam.pixel2normalized_vector(matches[0])
    bearings_frm = cam.pixel2normalized_vector(matches[1])
    # cam_gt = extend_SO3_to_homogenous(rot).dot(cam_gt).dot(extend_SO3_to_homogenous(rot.T))

    cam_8p = g8p.recover_pose_from_matches(x1=bearings_kf.copy(),
                                           x2=bearings_frm.copy())

    # # ! prior motion
    prior_motion = cam_8p[0:3, 3]
    rot = get_rot_from_directional_vectors(prior_motion, (1, 0, 1))
    bearings_a_rot = rot.dot(bearings_kf)
    bearings_b_rot = rot.dot(bearings_frm)
    #
    cam_a2b_n8p_rot = g8p_norm.recover_pose_from_matches(x1=bearings_a_rot.copy(),
                                                         x2=bearings_b_rot.copy())

    cam_8p_norm = extend_SO3_to_homogenous(rot.T).dot(cam_a2b_n8p_rot).dot(extend_SO3_to_homogenous(rot))

    # cam_8p_norm = g8p_norm.recover_pose_from_matches(x1=bearings_kf.copy(),
    #                                                  x2=bearings_frm.copy())

    error_8p.append(evaluate_error_in_transformation(transform_gt=cam_gt,
                                                     transform_est=cam_8p))
    error_n8p.append(evaluate_error_in_transformation(transform_gt=cam_gt,
                                                      transform_est=cam_8p_norm))

    # print("ours:  {}".format(error_n8p[-1]))
    # print("8PA:   {}".format(error_8p[-1]))
    # # print("kf:{} - frm:{} - matches:{}".format(tracker.initial_frame.idx,
    # #                                            tracker.tracked_frame.idx,
    # #                                            len(tracker.tracks)))
    # print("=====================================================================")
    # # cv2.waitKey(0)
    print("=====================================================================")
    # ! Ours' method
    print("Q1-ours:{}- {}".format(np.quantile(error_n8p, 0.25, axis=0),
                                  len(error_n8p)))
    print("Q2-ours:{}- {}".format(np.median(error_n8p, axis=0),
                                  len(error_n8p)))
    print("Q3-ours:{}- {}".format(np.quantile(error_n8p, 0.75, axis=0),
                                  len(error_n8p)))

    print("=====================================================================")
    # ! 8PA
    print("Q1-8PA:{}-  {}".format(np.quantile(error_8p, 0.25, axis=0),
                                  len(error_8p)))
    print("Q2-8PA:{}-  {}".format(np.median(error_8p, axis=0),
                                  len(error_8p)))
    print("Q3-8PA:{}-  {}".format(np.quantile(error_8p, 0.75, axis=0),
                                  len(error_8p)))


if __name__ == '__main__':
    error_n8p = []
    error_8p = []
    scene = "1LXtFkjw3qL/1"
    path = "/home/kike/Documents/datasets/Matterport_360_odometry"
    dt = MP3D_VO(scene=scene, path=path)
    orb = ORBExtractor()
    tracker = LKTracker()
    g8p_norm = g8p_norm()
    g8p = g8p()
    threshold_camera_distance = 0.1
    camera_distance = 0
    i = 0
    for idx in range(i, dt.number_frames, 1):
        frame_curr = Frame(**dt.get_frame(idx, return_dict=True))

        if idx == i:
            tracker.set_initial_frame(initial_frame=frame_curr, extractor=orb)
            continue

        relative_pose = frame_curr.get_relative_pose(key_frame=tracker.initial_frame)
        camera_distance = np.linalg.norm(relative_pose[0:3, 3])

        if camera_distance > threshold_camera_distance:
            eval_camera_pose(tracker=tracker, cam_gt=relative_pose)
            frame_prev = tracker.tracked_frame
            tracker.set_initial_frame(initial_frame=frame_prev, extractor=orb)
            relative_pose = frame_curr.get_relative_pose(key_frame=tracker.initial_frame)
            camera_distance = np.linalg.norm(relative_pose[0:3, 3])

        tracked_img = tracker.track(frame=frame_curr)
        # print("Camera Distance       {}".format(camera_distance))
        # print("Tracked features      {}".format(len(tracker.tracks)))
        # print("KeyFrame/CurrFrame:   {}-{}".format(tracker.initial_frame.idx, frame_curr.idx))
        cv2.imshow("preview", tracked_img[:, :, ::-1])
        cv2.waitKey(10)
