import cv2
from structures.frame import Frame
import pandas as pd
import numpy as np
from sphere import Sphere
from solvers.epipolar_constraint_by_ransac import RansacEssentialMatrix
from pcl_utilities import *
from geometry_utilities import *
from solvers.epipolar_constraint import EightPointAlgorithmGeneralGeometry as g8p
import os
from file_utilities import save_obj, load_obj
from image_utilities import get_mask_map_by_res_loc
from analysis.utilities.camera_recovering import *
from file_utilities import create_dir

colors = dict(
    COLOR_NORM='rgb(0,200,200)',
    COLOR_HARTLEY_8PA='rgb(0,50,255)',
    COLOR_8PA='rgb(30,144,255)',
    COLOR_OURS_NORM_8PA='rgb(255,127,80)',
    COLOR_OPT_RT='rgb(0,255,100)',
    COLOR_OURS_OPT_RES_RT='rgb(0,100,80)',
    COLOR_GENERAL='red',
)


def sampling_bearings(**kwargs):
    if "sampling" in kwargs.keys():
        number_of_samples = kwargs["bearings"]["kf"].shape[1]
        samples = np.random.randint(0, number_of_samples, kwargs.get("sampling", number_of_samples))
        kwargs["bearings"]["kf"] = kwargs["bearings"]["kf"][:, samples]
        kwargs["bearings"]["frm"] = kwargs["bearings"]["frm"][:, samples]
    return kwargs


def get_bearings(**kwargs):
    bearings_kf, bearings_frm, cam_gt, kwargs, ret = track_features(**kwargs)
    if not ret:
        return kwargs, False
    # ! 8PA Evaluation
    kwargs["bearings"] = dict()
    if kwargs.get("use_ransac", False):
        # ! Solving by using RANSAC
        ransac = RansacEssentialMatrix(**kwargs)
        _ = ransac.solve(data=(bearings_kf.copy().T, bearings_frm.copy().T))
        num_inliers = sum(ransac.current_inliers)
        num_of_samples = len(ransac.current_inliers)
        kwargs["bearings"]["rejections"] = 1 - (num_inliers / num_of_samples)
        kwargs["bearings"]["ransac_residuals"] = ransac.current_residual
        kwargs["bearings"]["kf"] = bearings_kf[:, ransac.current_inliers]
        kwargs["bearings"]["frm"] = bearings_frm[:, ransac.current_inliers]
    else:
        kwargs["bearings"]["kf"] = bearings_kf
        kwargs["bearings"]["frm"] = bearings_frm

    kwargs = sampling_bearings(**kwargs)
    kwargs["cam_gt"] = cam_gt
    kwargs["e_gt"] = g8p.get_e_from_cam_pose(cam_gt)
    if kwargs.get("timing_evaluation", False):
        cam_pose, _, _ = get_cam_pose_by_8pa(**kwargs)
    else:
        cam_pose, _ = get_cam_pose_by_8pa(**kwargs)

    kwargs["landmarks_kf"] = g8p.triangulate_points_from_cam_pose(
        cam_pose=cam_pose,
        x1=kwargs["bearings"]['kf'].copy(),
        x2=kwargs["bearings"]['frm'].copy(),
    )
    if kwargs.get("save_bearings", False):
        save_bearings(**kwargs)
    print("Number of pts: {}".format(kwargs["bearings"]["kf"].shape[1]))
    return kwargs, ret


def eval_cam_pose_error(_print=True, **kwargs):
    cams = [cam for cam in kwargs.keys() if "cam" in cam and "gt" not in cam]
    cam_gt = kwargs["cam_gt"]
    for cam in cams:
        cam_pose = kwargs[cam]
        error_name = "error_" + cam
        error = evaluate_error_in_transformation(
            transform_est=cam_pose, transform_gt=cam_gt)
        if error_name + "_rot" not in kwargs["results"].keys():
            kwargs["results"][error_name + "_rot"] = [error[0]]
            kwargs["results"][error_name + "_tran"] = [error[1]]
        else:
            kwargs["results"][error_name + "_rot"].append(error[0])
            kwargs["results"][error_name + "_tran"].append(error[1])
        if _print:
            print("--------------------------------------------------------")
            print("solver:{}".format(cam))
            # print("75% Error-rot: {}".format(np.quantile(kwargs["results"][error_name + "_rot"], 0.75)))
            print("50% Error-rot: {}".format(
                np.quantile(kwargs["results"][error_name + "_rot"], 0.5)))
            # print("25% Error-rot: {}".format(np.quantile(kwargs["results"][error_name + "_rot"], 0.25)))
            # print("--------------------------------------------------------")
            # print("75% Error-tran: {}".format(np.quantile(kwargs["results"][error_name + "_tran"], 0.75)))
            print("50% Error-tran: {}".format(
                np.quantile(kwargs["results"][error_name + "_tran"], 0.5)))
            # print("25% Error-tran: {}".format(np.quantile(kwargs["results"][error_name + "_tran"], 0.25)))
            # print("--------------------------------------------------------")

    losses = [loss for loss in kwargs.keys() if "loss" in loss]
    for loss in losses:
        ls = kwargs[loss].copy()

        if loss not in kwargs["results"].keys():
            kwargs["results"][loss] = [ls]
        else:
            kwargs["results"][loss].append(ls)

    if kwargs.get("timing_evaluation", False):
        time_evaluation = [time_ for time_ in kwargs.keys() if "time" in time_]
        print("*****************************************")
        for eval in time_evaluation:
            if eval not in kwargs["results"].keys():
                kwargs["results"][eval] = [kwargs[eval]]
            else:
                kwargs["results"][eval].append(kwargs[eval])

            if _print:
                print("MED time {}: {}".format(eval,
                                               np.quantile(kwargs["results"][eval], 0.5)))
    print("done")
    return kwargs


def error_eval(**kwargs):
    cams = [cam for cam in kwargs.keys() if "cam" in cam and "gt" not in cam]
    for cam in cams:
        error_name = "error_" + cam
        error = (np.nan, np.nan)
        if error_name + "_rot" not in kwargs["results"].keys():
            kwargs["results"][error_name + "_rot"] = [error[0]]
            kwargs["results"][error_name + "_tran"] = [error[1]]
        else:
            kwargs["results"][error_name + "_rot"].append(error[0])
            kwargs["results"][error_name + "_tran"].append(error[1])

    losses = [loss for loss in kwargs.keys() if "loss" in loss]
    for loss in losses:
        ls = np.nan

        if loss not in kwargs["results"].keys():
            kwargs["results"][loss] = [ls]
        else:
            kwargs["results"][loss].append(ls)

    if kwargs.get("timing_evaluation", False):
        time_evaluation = [time_ for time_ in kwargs.keys() if "time" in time_]
        for eval in time_evaluation:
            if eval not in kwargs["results"].keys():
                kwargs["results"][eval] = [np.nan]
            else:
                kwargs["results"][eval].append(np.nan)
    return kwargs


def msk(eval, quantile):
    pivot = np.quantile(eval, quantile)
    # pivot = np.inf
    mask = eval > pivot
    eval[mask] = pivot
    return eval


def save_bearings_vectors(**kwargs):
    while True:
        bearings_kf, bearings_frm, cam_gt, kwargs, ret = track_features(**kwargs)

        if not ret:
            break

        kwargs["bearings"] = dict()
        kwargs["bearings"]["kf"] = bearings_kf
        kwargs["bearings"]["frm"] = bearings_frm
        save_bearings(**kwargs)


def track_features(**kwargs):
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
            continue

        relative_pose = frame_curr.get_relative_pose(
            key_frame=kwargs["tracker"].initial_frame)
        camera_distance = np.linalg.norm(relative_pose[0:3, 3])

        tracked_img = kwargs["tracker"].track(frame=frame_curr)
        if kwargs["show_tracked_features"]:
            print("Camera Distance       {}".format(camera_distance))
            print("Tracked features      {}".format(
                len(kwargs["tracker"].tracks)))
            print("KeyFrame/CurrFrame:   {}-{}".format(
                kwargs["tracker"].initial_frame.idx, frame_curr.idx))
            cv2.imshow("preview", tracked_img[:, :, ::-1])
            cv2.waitKey(10)

        if camera_distance > kwargs.get("distance_threshold", 0.5):
            break
        idx += 1
        if not idx < kwargs["data_scene"].number_frames:
            break

    relative_pose = frame_curr.get_relative_pose(
        key_frame=kwargs["tracker"].initial_frame)

    # ! Maybe for different camera projection we want to change this
    # ! However, we can consider every projection as spherical one
    cam = kwargs["data_scene"].cam
    # cam = Sphere(shape=kwargs["tracker"].initial_frame.shape)
    matches = kwargs["tracker"].get_matches()
    kwargs["results"]["kf"].append(kwargs["tracker"].initial_frame.idx)
    try:
        bearings_kf = cam.pixel2euclidean_space(matches[0])
        bearings_frm = cam.pixel2euclidean_space(matches[1])
    except:
        bearings_kf = None
        bearings_frm = None
        print("error")
    if kwargs.get("special_eval", False):
        kwargs["idx_frame"] += 1
    else:
        kwargs["idx_frame"] = kwargs["tracker"].frame_idx

    return bearings_kf, bearings_frm, relative_pose, kwargs, ret


def save_bearings(**kwargs):
    if kwargs["bearings"]["kf"] is not None:
        dt = pd.DataFrame(np.vstack((kwargs["bearings"]["kf"], kwargs["bearings"]["frm"])).T)
        dirname = os.path.join(os.path.dirname(kwargs["filename"]), "frames")
        file_bearings = str(kwargs["tracker"].initial_frame.idx) + "_" + str(
            kwargs["tracker"].tracked_frame.idx) + ".txt"
        file_bearings = os.path.join(dirname, file_bearings)
        print("scene:{}".format(kwargs["data_scene"].scene))
        print("Frames Kf:{}-frm:{}".format(kwargs["tracker"].initial_frame.idx, kwargs["tracker"].tracked_frame.idx))
        print("tracked features {}".format(kwargs["bearings"]["kf"].shape[1]))
        create_dir(dirname, delete_previous=False)
        dt.to_csv(file_bearings, header=None, index=None)


def get_bearings_by_plc(**kwargs):
    # ! Getting a PCL from the dataset
    ret = True
    idx_frame = kwargs["idx_frame"]
    loc = kwargs["loc"]
    res = kwargs["res"]
    pcl_dense, pcl_dense_color, _, _ = kwargs["data_scene"].get_pcl(idx=idx_frame)
    pcl_dense, mask = mask_pcl_by_res_and_loc(pcl=pcl_dense, loc=loc, res=res)

    linear_motion = kwargs.get("linear_motion", (-1, 1))
    angular_motion = kwargs.get("angular_motion", (-10, 10))

    cam_a2b = get_homogeneous_transform_from_vectors(
        t_vector=(np.random.uniform(linear_motion[0], linear_motion[1]),
                  np.random.uniform(linear_motion[0], linear_motion[1]),
                  np.random.uniform(linear_motion[0], linear_motion[1])),
        r_vector=(np.random.uniform(angular_motion[0], angular_motion[1]),
                  np.random.uniform(angular_motion[0], angular_motion[1]),
                  np.random.uniform(angular_motion[0], angular_motion[1])))

    pts = kwargs.get("sampling", 200)
    samples = np.random.randint(0, pcl_dense.shape[1], pts)
    pcl_a = extend_array_to_homogeneous(pcl_dense[:, samples])

    pcl_b = add_noise_to_pcl(np.linalg.inv(cam_a2b).dot(pcl_a),
                             param=kwargs.get("noise", 500))

    pcl_b = add_outliers_to_pcl(
        pcl_b.copy(),
        inliers=int(kwargs.get("inliers_ratio", 0.5) * pts))

    bearings_a = sph.sphere_normalization(pcl_a)
    bearings_b = sph.sphere_normalization(pcl_b)

    if "results" not in kwargs.keys():
        kwargs["results"] = dict()
        kwargs["results"]["kf"] = [idx_frame]
    else:
        kwargs["results"]["kf"].append(idx_frame)

    if "skip_frames" in kwargs.keys():
        kwargs["idx_frame"] += kwargs["skip_frames"]
    else:
        kwargs["idx_frame"] += 1
    if kwargs["idx_frame"] >= kwargs["data_scene"].number_frames:
        ret = False

    kwargs["bearings"] = dict()
    kwargs["bearings"]["kf"] = bearings_a
    kwargs["bearings"]["frm"] = bearings_b
    kwargs["cam_gt"] = cam_a2b
    return kwargs, ret
