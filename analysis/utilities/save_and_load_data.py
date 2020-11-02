import pandas as pd
from analysis.utilities.data_utilities import *
from file_utilities import *


def load_bearings_from_sampled_pcl(**kwargs):
    if "saved_bearings_files" not in kwargs.keys():
        dir_result = os.path.join(os.path.dirname(os.path.dirname(kwargs["filename"])), "saved_bearings")
        saved_frames_dir = [d for d in os.listdir(dir_result) if "saving_bearings_from_sampled_pcl" in d]
        saved_frames_dir = [d for d in saved_frames_dir if kwargs["keyword"] in d][0]
        list_saved_files = [f for f in os.listdir(os.path.join(dir_result, saved_frames_dir)) if
                            f not in ("cam_poses.txt", "settings.config")]

        list_saved_files.sort(key=lambda x: int(os.path.splitext(x)[0]))
        list_saved_files = [os.path.join(dir_result, saved_frames_dir, f) for f in list_saved_files]
        kwargs["saved_bearings_files"] = list_saved_files
        kwargs["poses"] = pd.read_csv(os.path.join(dir_result, saved_frames_dir, "cam_poses.txt"),
                                      header=None).values
        kwargs["results"] = dict()
        kwargs["idx_eval"] = 0
    else:
        kwargs["idx_eval"] += 1
        if len(kwargs["saved_bearings_files"]) < kwargs["idx_eval"] + 1:
            return kwargs, False

    idx = kwargs["idx_eval"]
    bearings_read = pd.read_csv(kwargs["saved_bearings_files"][idx], header=None).values
    if bearings_read.shape[0] < 100:
        return kwargs, True

    cam_gt = kwargs["poses"][idx, :].reshape((4, 4))
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

    kwargs["landmarks_kf"] = g8p().triangulate_points_from_cam_pose(
        cam_pose=cam_pose,
        x1=kwargs["bearings"]['kf'].copy(),
        x2=kwargs["bearings"]['frm'].copy(),
    )
    return kwargs, True


def load_bearings_from_tracked_features(**kwargs):
    if "saved_bearings_files" not in kwargs.keys():
        dir_result = os.path.join(os.path.dirname(os.path.dirname(kwargs["filename"])), "saved_bearings")
        saved_frames_dir = [d for d in os.listdir(dir_result) if kwargs["keyword"] in d][0]
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
    bearings_read = pd.read_csv(kwargs["saved_bearings_files"][idx], header=None).values
    if bearings_read.shape[0] < 100:
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

    kwargs["landmarks_kf"] = g8p().triangulate_points_from_cam_pose(
        cam_pose=cam_pose,
        x1=kwargs["bearings"]['kf'].copy(),
        x2=kwargs["bearings"]['frm'].copy(),
    )
    return kwargs, True


def save_bearings_vectors(**kwargs):
    from analysis.utilities.data_utilities import track_features as trk
    from analysis.utilities.data_utilities import sampling_bearings as sampling

    while True:
        bearings_kf, bearings_frm, cam_gt, kwargs, ret = trk(**kwargs)

        if not ret:
            break

        kwargs["bearings"] = dict()
        kwargs["bearings"]["kf"] = bearings_kf
        kwargs["bearings"]["frm"] = bearings_frm
        if "sampling" in kwargs.keys():
            kwargs = sampling(**kwargs)

        save_bearings(**kwargs)


def save_bearings(**kwargs):
    if kwargs["bearings"]["kf"] is not None:
        dt = pd.DataFrame(np.vstack((kwargs["bearings"]["kf"], kwargs["bearings"]["frm"])).T)
        dirname = os.path.join(os.path.dirname(os.path.dirname(kwargs["filename"])),
                               "saved_bearings",
                               "saving_tracked_features" +
                               "_samples_" + str(kwargs["sampling"]) +
                               "_dist_" + str(kwargs["distance_threshold"]) +
                               "_res_" + str(kwargs["res"][0]) + "." + str(kwargs["res"][1]) +
                               "_loc_" + str(kwargs["loc"][0]) + "." + str(kwargs["loc"][1]),
                               "frames",
                               )
        file_bearings = str(kwargs["tracker"].initial_frame.idx) + "_" + str(
            kwargs["tracker"].tracked_frame.idx) + ".txt"
        file_bearings = os.path.join(dirname, file_bearings)
        print("scene:{}".format(kwargs["data_scene"].scene))
        print("Frames Kf:{}-frm:{}".format(kwargs["tracker"].initial_frame.idx, kwargs["tracker"].tracked_frame.idx))
        print("tracked features {}".format(kwargs["bearings"]["kf"].shape[1]))
        create_dir(dirname, delete_previous=False)
        print(file_bearings)
        dt.to_csv(file_bearings, header=None, index=None)


def save_data(**kwargs):
    filename = kwargs["filename"] + ".txt"
    headers = [r for r in kwargs["results"]]
    if "saved_list" in kwargs.keys():
        headers.extend([r for r in kwargs["saved_list"]])
    if not os.path.isfile(filename):
        write_report(filename, headers, flag="w+")
    line = []
    for key in kwargs["results"]:
        line.append(kwargs["results"][key][-1])
    if "saved_list" in kwargs.keys():
        for key in kwargs["saved_list"]:
            line.append(kwargs[key])
    write_report(filename, line, flag="a")
