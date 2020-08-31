import cv2
from structures.frame import Frame
import numpy as np
from sphere import Sphere


def track_features(**kwargs):
    initial_frame = kwargs["idx_frame"]
    idx = initial_frame
    while True:
        frame_curr = Frame(**kwargs["data_scene"].get_frame(idx, return_dict=True),
                           **dict(idx=idx))
        if idx == initial_frame:
            kwargs["tracker"].set_initial_frame(initial_frame=frame_curr,
                                                extractor=kwargs["feat_extractor"],
                                                mask=kwargs["mask"])
            idx += 1
            continue
        idx += 1
        relative_pose = frame_curr.get_relative_pose(
            key_frame=kwargs["tracker"].initial_frame)
        camera_distance = np.linalg.norm(relative_pose[0:3, 3])

        tracked_img = kwargs["tracker"].track(frame=frame_curr)
        if kwargs["show_tracked_features"]:
            print("Camera Distance       {}".format(camera_distance))
            print("Tracked features      {}".format(len(kwargs["tracker"].tracks)))
            print("KeyFrame/CurrFrame:   {}-{}".format(kwargs["tracker"].initial_frame.idx, frame_curr.idx))
            cv2.imshow("preview", tracked_img[:, :, ::-1])
            cv2.waitKey(10)

        if camera_distance > kwargs["distance_threshold"]:
            break

    relative_pose = frame_curr.get_relative_pose(
        key_frame=kwargs["tracker"].initial_frame)

    # ! Maybe for different camera projection we want to change this
    # ! However, we can consider every projection as spherical one

    cam = Sphere(shape=kwargs["tracker"].initial_frame.shape)
    matches = kwargs["tracker"].get_matches()
    bearings_kf = cam.pixel2normalized_vector(matches[0])
    bearings_frm = cam.pixel2normalized_vector(matches[1])
    return bearings_kf, bearings_frm, relative_pose, kwargs
