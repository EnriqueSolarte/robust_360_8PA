import os
import numpy as np
from image_utilities import get_mask_map_by_res_loc
from structures.frame import Frame
import cv2
from read_datasets.MP3D_VO import MP3D_VO
from structures.tracker import LKTracker
from structures.extractor.shi_tomasi_extractor import Shi_Tomasi_Extractor


def record_features(**kwargs):
    # fourcc = cv2.VideoWriter_fourcc(*'XVID')
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')

    out = cv2.VideoWriter('{}_{}.mp4'.format(kwargs["extra"],
                                             kwargs["data_scene"].scene), fourcc, 15.0,
                          (kwargs["data_scene"].shape[1], kwargs["data_scene"].shape[0]))

    while True:
        # ! It stops at the end of the sequence
        if not kwargs["idx_frame"] + 1 < kwargs["data_scene"].number_frames:
            return None, None, None, kwargs, False
        if "results" not in kwargs.keys():
            kwargs["results"] = dict()
            kwargs["results"]["kf"] = []

        if kwargs.get("mask_in_all_image", False):
            kwargs["mask"] = np.ones(kwargs["data_scene"].shape).astype(np.uint8)
        else:
            if 'mask' not in kwargs.keys():
                kwargs["mask"] = get_mask_map_by_res_loc(
                    kwargs["data_scene"].shape,
                    res=kwargs["res"],
                    loc=kwargs["loc"])

        initial_frame = kwargs["idx_frame"]
        idx = initial_frame
        ret = True

        while True:
            frame_curr = Frame(
                **kwargs["data_scene"].get_frame(idx, return_dict=True),
                **dict(idx=idx))
            if idx == initial_frame:
                kwargs["tracker"].set_initial_frame(
                    initial_frame=frame_curr,
                    extractor=kwargs["feat_extractor"],
                    mask=kwargs["mask"])
                idx += 1
                kwargs["tracker"].aux_indexes = np.random.randint(
                    low=0,
                    high=len(kwargs["tracker"].tracks),
                    size=int(len(kwargs["tracker"].tracks) * np.random.uniform(0.05, 0.20, 1)))
                continue

            relative_pose = frame_curr.get_relative_pose(
                key_frame=kwargs["tracker"].initial_frame)
            camera_distance = np.linalg.norm(relative_pose[0:3, 3])

            tracked_img = kwargs["tracker"].track(frame=frame_curr)
            print("Tracked features      {}".format(
                len(kwargs["tracker"].tracks)))
            cv2.imshow("preview", tracked_img[:, :, ::-1])
            out.write(tracked_img[:, :, ::-1])
            cv2.waitKey(1)

            if camera_distance > kwargs.get("distance_threshold", 0.5):
                break
            idx += 1
            if not idx < kwargs["data_scene"].number_frames:
                break

        relative_pose = frame_curr.get_relative_pose(
            key_frame=kwargs["tracker"].initial_frame)

        kwargs["results"]["kf"].append(kwargs["tracker"].initial_frame.idx)
        if kwargs.get("special_eval", False):
            kwargs["idx_frame"] += 1
        else:
            kwargs["idx_frame"] = kwargs["tracker"].frame_idx

        if not ret:
            break
    out.release()
