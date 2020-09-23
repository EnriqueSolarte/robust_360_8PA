import cv2
import numpy as np

from read_datasets.KITTI import KITTI_VO
from sphere import Sphere
from pinhole import Pinhole
from structures.frame import Frame
from solvers.epipolar_constraint_by_ransac import RansacEssentialMatrix


def BFMatcher(img1, img2):
    img1 = img1.copy()
    img2 = img2.copy()

    # Initiate SIFT detector
    sift = cv2.xfeatures2d.SIFT_create()

    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)
    # BFMatcher with default params
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)
    # Apply ratio test
    matchesMask = np.array([0 for i in range(len(matches))])
    good = []
    for i, (m, n) in enumerate(matches):
        if 0.50 * n.distance < m.distance < 0.85 * n.distance:
            good.append(m)
            matchesMask[i] = 1
    src_pts = [tuple([int(pos) for pos in kp1[m.queryIdx].pt]) for m in good]
    dst_pts = [tuple([int(pos) for pos in kp2[m.trainIdx].pt]) for m in good]
    # return dict(zip(src_pts, dst_pts))
    return [src_pts, dst_pts]


def eval_camera_pose(initial_frame, key_frame, cam_gt):
    from solvers.epipolar_constraint import EightPointAlgorithmGeneralGeometry as g8p
    from solvers.optimal8pa import Optimal8PA as norm_8pa

    g8p_norm = norm_8pa(version=opt_version)
    g8p = g8p()

    matches = np.array(BFMatcher(initial_frame.color_map, key_frame.color_map))

    cam = Pinhole()
    kf = cam.pixel2normalized_vector(matches[0])
    frm = cam.pixel2normalized_vector(matches[1])

    ransac_parm = dict(
        min_samples=8,
        p_succes=0.99,
        outliers=0.5,
        residual_threshold=0.01,
        verbose=True)

    print("Number of matches: {}".format(kf.shape[1]))
    cam_8p = RansacEssentialMatrix(**ransac_parm).solve(
        data=(np.copy(kf.T), np.copy(frm.T)), solver="g8p")
    cam_n8p = RansacEssentialMatrix(**ransac_parm).solve(
        data=(np.copy(kf.T), np.copy(frm.T)), solver="norm_8pa")


if __name__ == "__main__":
    from config import *

    if dataset == "kitti":
        data = KITTI_VO(basedir=basedir, scene=scene)

    i = 0
    threshold_camera_distance = 5

    for idx in range(i, data.number_frames):
        frame_curr = Frame(**data.get_frame(idx, return_dict=True), idx=idx)

        if idx == i:
            initial_frame = frame_curr
            continue

        relative_pose = frame_curr.get_relative_pose(key_frame=initial_frame)
        camera_distance = np.linalg.norm(relative_pose[0:3, 3])
        print("Distance: " + str(camera_distance))

        if camera_distance > threshold_camera_distance:
            eval_camera_pose(
                initial_frame=initial_frame,
                key_frame=frame_curr,
                cam_gt=relative_pose)
            frame_prev = frame_curr
            initial_frame = frame_prev
            relative_pose = frame_curr.get_relative_pose(
                key_frame=initial_frame)
            camera_distance = np.linalg.norm(relative_pose[0:3, 3])

        cv2.namedWindow("preview")
        cv2.imshow("preview", frame_curr.color_map)
        cv2.waitKey(1)
