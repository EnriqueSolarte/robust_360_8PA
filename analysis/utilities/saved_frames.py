import os
import pandas
import numpy as np
from geometry_utilities import *
from analysis.utilities.data_utilities import *


def load_bearings(**kwargs):
    if "saved_bearings_files" not in kwargs.keys():
        dir_result = os.path.dirname(os.path.dirname(kwargs["filename"]))
        saved_frames_dir = [d for d in os.listdir(dir_result) if "saving_tracked_features" in d][0]
        list_saved_files = os.listdir(os.path.join(dir_result, saved_frames_dir + "/frames"))
        list_saved_files = [os.path.join(dir_result, saved_frames_dir + "/frames", f) for f in list_saved_files]
        kwargs["saved_bearings_files"] = list_saved_files
        kwargs["results"] = dict()
        kwargs["idx_eval"] = 0
    else:
        kwargs["idx_eval"] += 1
        if len(kwargs["saved_bearings_files"]) < kwargs["idx_eval"] + 1:
            return kwargs, False

    idx = kwargs["idx_eval"]
    bearings_read = pandas.read_csv(kwargs["saved_bearings_files"][idx]).values
    if bearings_read.shape[0] < 8:
        return kwargs, True

    file_ = os.path.splitext(os.path.split(kwargs["saved_bearings_files"][idx])[1])[0]
    frames = [int(f) for f in file_.split("_")]
    print("idx :{} frames: {}".format(idx, frames))
    frame_kf = kwargs["data_scene"].get_frame(frames[0], return_dict=True)
    frame_frm = kwargs["data_scene"].get_frame(frames[1], return_dict=True)

    cam_gt = np.linalg.inv(frame_kf["pose"]).dot(frame_frm["pose"])
    bearings_kf = bearings_read[:, 0:3].T
    bearings_frm = bearings_read[:, 3:6].T
    kwargs["bearings"] = dict()
    kwargs["bearings"]["kf"] = bearings_kf
    kwargs["bearings"]["frm"] = bearings_frm
    kwargs["cam_gt"] = cam_gt
    try:
        if kwargs.get("timing_evaluation", False):
            cam_pose, _, _ = get_cam_pose_by_8pa(**kwargs)
        else:
            cam_pose, _ = get_cam_pose_by_8pa(**kwargs)
    except:
        return kwargs, True

    kwargs["landmarks_kf"] = g8p.triangulate_points_from_cam_pose(
        cam_pose=cam_pose,
        x1=kwargs["bearings"]['kf'].copy(),
        x2=kwargs["bearings"]['frm'].copy(),
    )
    return kwargs, True
