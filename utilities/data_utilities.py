import cv2
from structures.frame import Frame
import numpy as np
from sphere import Sphere
from solvers.epipolar_constraint_by_ransac import RansacEssentialMatrix
from pcl_utilities import *
from geometry_utilities import *
from solvers.epipolar_constraint import EightPointAlgorithmGeneralGeometry as g8p
import os
from file_utilities import save_obj, load_obj
from image_utilities import get_mask_map_by_res_loc

COLOR_8PA = 'rgb(30,144,255)'
COLOR_NORM_8PA = 'rgb(255,127,80)'


def get_bearings(**kwargs):
    bearings_kf, bearings_frm, cam_gt, kwargs, ret = track_features(**kwargs)
    # ! 8PA Evaluation
    kwargs["bearings"] = dict()

    if kwargs.get("use_ransac", False):
        # ! Solving by using RANSAC
        ransac = RansacEssentialMatrix(**kwargs)
        _ = ransac.solve(data=(
            bearings_kf.copy().T,
            bearings_frm.copy().T)
        )
        num_inliers = sum(ransac.current_inliers)
        num_of_samples = len(ransac.current_inliers)
        kwargs["bearings"]["rejections"] = 1 - (num_inliers / num_of_samples)
        kwargs["bearings"]["ransac_residuals"] = ransac.current_residual
        kwargs["bearings"]["kf"] = bearings_kf[:, ransac.current_inliers]
        kwargs["bearings"]["frm"] = bearings_frm[:, ransac.current_inliers]
    else:
        kwargs["bearings"]["kf"] = bearings_kf
        kwargs["bearings"]["frm"] = bearings_frm

    kwargs["cam_gt"] = cam_gt
    return kwargs


def single_eval_cam_pose_error(_print=True, **kwargs):
    cams = [cam for cam in kwargs.keys() if "cam" in cam and "gt" not in cam]
    cam_gt = kwargs["cam_gt"]
    for cam in cams:
        cam_pose = kwargs[cam]
        kwargs["error_" + cam] = evaluate_error_in_transformation(
            transform_est=cam_pose,
            transform_gt=cam_gt
        )
        if _print:
            print("{} Error-rot: {}".format(cam, kwargs["error_" + cam][0]))
            print("{} Error-tran: {}".format(cam, kwargs["error_" + cam][1]))
    return kwargs


def msk(eval, quantile):
    pivot = np.quantile(eval, quantile)
    # pivot = np.inf
    mask = eval > pivot
    eval[mask] = pivot
    return eval


def save_results(**kwargs):
    filename = kwargs["filename"]
    dir_output = os.path.join("plots/{}.data".format(filename))
    save_obj(dir_output, kwargs["results"])


def save_surface_results(**kwargs):
    filename = kwargs["filename"]
    dir_output = os.path.join("plots/{}.data".format(filename))

    dt = dict(results=kwargs["results"],
              v_grid=kwargs["v_grid"],
              vv_grid=kwargs["vv_grid"])
    save_obj(dir_output, dt)


def track_features(**kwargs):
    if not kwargs["tracker"].frame_idx + 1 < kwargs["data_scene"].number_frames:
        return None, None, None, kwargs, False

    kwargs["mask"] = get_mask_map_by_res_loc(kwargs["data_scene"].shape,
                                             res=kwargs["res"],
                                             loc=kwargs["loc"])
    initial_frame = kwargs["idx_frame"]
    idx = initial_frame
    ret = True

    while True:
        frame_curr = Frame(**kwargs["data_scene"].get_frame(idx, return_dict=True),
                           **dict(idx=idx))
        if idx == initial_frame:
            kwargs["tracker"].set_initial_frame(initial_frame=frame_curr,
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
            print("Tracked features      {}".format(len(kwargs["tracker"].tracks)))
            print("KeyFrame/CurrFrame:   {}-{}".format(kwargs["tracker"].initial_frame.idx, frame_curr.idx))
            cv2.imshow("preview", tracked_img[:, :, ::-1])
            cv2.waitKey(10)

        if camera_distance > kwargs["distance_threshold"]:
            break
        idx += 1
        if not idx < kwargs["data_scene"].number_frames:
            break

    relative_pose = frame_curr.get_relative_pose(
        key_frame=kwargs["tracker"].initial_frame)

    # ! Maybe for different camera projection we want to change this
    # ! However, we can consider every projection as spherical one

    cam = Sphere(shape=kwargs["tracker"].initial_frame.shape)
    matches = kwargs["tracker"].get_matches()
    bearings_kf = cam.pixel2normalized_vector(matches[0])
    bearings_frm = cam.pixel2normalized_vector(matches[1])
    kwargs["idx_frame"] = kwargs["tracker"].frame_idx

    return bearings_kf, bearings_frm, relative_pose, kwargs, ret