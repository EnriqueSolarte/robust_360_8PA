from read_datasets.MP3D_VO import MP3D_VO

import cv2
from extractor import ORBExtractor
from tracker import LKTracker
from frame import Frame
from geometry_utilities import *
from file_utilities import create_dir, write_report, create_file

from config import *

error_n8p, error_8p = [], []


def eval_camera_pose(tracker, cam_gt, output_dir, file):
    # ! relative camera pose from a to b

    from solvers.epipolar_constraint import EightPointAlgorithmGeneralGeometry as g8p
    from solvers.optimal8pa import Optimal8PA as norm_8pa

    g8p_norm = norm_8pa(version=opt_version)
    g8p = g8p()

    cam = Sphere(shape=tracker.initial_frame.shape)
    matches = tracker.get_matches()
    # rot = get_rot_from_directional_vectors((0, 0, 1), (0, 0, 1))
    bearings_kf = cam.pixel2normalized_vector(matches[0])
    bearings_frm = cam.pixel2normalized_vector(matches[1])

    # cam_gt = extend_SO3_to_homogenous(rot).dot(cam_gt).dot(extend_SO3_to_homogenous(rot.T))

    cam_8p = g8p.recover_pose_from_matches(x1=bearings_kf.copy(),
                                           x2=bearings_frm.copy())

    if motion_constraint:
        # ! Forward motion constraint
        prior_motion = cam_8p[0:3, 3]
        rot = get_rot_from_directional_vectors(prior_motion, (0, 0, 1))
        bearings_a_rot = rot.dot(bearings_kf)
        bearings_b_rot = rot.dot(bearings_frm)

        cam_n8p_rot = g8p_norm.recover_pose_from_matches(
            x1=bearings_a_rot.copy(), x2=bearings_b_rot.copy())

        cam_n8p = extend_SO3_to_homogenous(rot.T).dot(cam_n8p_rot).dot(
            extend_SO3_to_homogenous(rot))
    else:
        cam_n8p = g8p_norm.recover_pose_from_matches(x1=bearings_kf.copy(),
                                                     x2=bearings_frm.copy())

        s1 = g8p_norm.T1[0][0]
        k1 = g8p_norm.T1[2][2]
        print("s1, k1 = ({}, {})".format(s1, k1))

        if opt_version != "v1":
            s2 = g8p_norm.T2[0][0]
            k2 = g8p_norm.T2[2][2]
            print("s2, k2 = ({}, {})".format(s2, k2))

    error_n8p.append(
        evaluate_error_in_transformation(transform_gt=cam_gt,
                                         transform_est=cam_n8p))
    error_8p.append(
        evaluate_error_in_transformation(transform_gt=cam_gt,
                                         transform_est=cam_8p))

    print("ours:  {}".format(error_n8p[-1]))
    print("8PA:   {}".format(error_8p[-1]))
    print("kf:{} - frm:{} - matches:{}".format(tracker.initial_frame.idx,
                                               tracker.tracked_frame.idx,
                                               len(tracker.tracks)))
    print(
        "====================================================================="
    )
    # cv2.waitKey(0)
    '''
        line = [
            error_n8p[-1][0],
            error_8p[-1][0],
            error_n8p[-1][1],
            error_8p[-1][1],
        ]
        write_report(os.path.join(output_dir, file), line)
    '''


if __name__ == '__main__':
    if dataset == "minos":
        data = MP3D_VO(basedir=path, scene=scene)

    orb = ORBExtractor(nfeatures=point)
    tracker = LKTracker()
    threshold_camera_distance = 0.5
    camera_distance = 0
    i = 0
    '''
    output_dir = os.path.join(OUTPUT_MP3D_SAMPLES, "camera_pose", scene, path)
    create_dir(output_dir)

    # ! Output file
    file = "camera_pose_estimation.csv"

    # ! Print fieldnames
    authors = ["Ours", "8PA"]
    oris = ["rot", "trans"]
    terms = ["Q1", "Q2", "Q3", "Mean", "STD"]

    fieldnames = []
    for ori in oris:
        for author in authors:
            fieldnames.append(author + "-" + ori)

    create_file(os.path.join(output_dir, file))
    write_report(os.path.join(output_dir, file), fieldnames)
    '''

    output_dir = None
    file = None

    mask = None

    # data.number_frames
    for idx in range(i, 200):
        frame_curr = Frame(**data.get_frame(idx, return_dict=True))

        if idx == i:
            mask = np.zeros_like(frame_curr.grey_map)
            h, w = mask.shape

            # ! Fov
            fov = (int(res[1] * h / 180), int(res[0] * w / 360))
            mask[h // 2 - fov[0] // 2:h // 2 + fov[0] // 2,
                 w // 2 - fov[1] // 2:w // 2 + fov[1] // 2] = 255

            tracker.set_initial_frame(initial_frame=frame_curr,
                                      extractor=orb,
                                      mask=mask)
            continue

        relative_pose = frame_curr.get_relative_pose(
            key_frame=tracker.initial_frame)
        camera_distance = np.linalg.norm(relative_pose[0:3, 3])

        if camera_distance > threshold_camera_distance:
            eval_camera_pose(tracker=tracker,
                             cam_gt=relative_pose,
                             output_dir=output_dir,
                             file=file)
            frame_prev = tracker.tracked_frame
            tracker.set_initial_frame(initial_frame=frame_prev,
                                      extractor=orb,
                                      mask=mask)
            relative_pose = frame_curr.get_relative_pose(
                key_frame=tracker.initial_frame)
            camera_distance = np.linalg.norm(relative_pose[0:3, 3])

        tracked_img = tracker.track(frame=frame_curr)
        # print("Camera Distance       {}".format(camera_distance))
        # print("Tracked features      {}".format(len(tracker.tracks)))
        # prinerror_8p[-1][0],t("KeyFrame/CurrFrame:   {}-{}".format(tracker.initial_frame.idx, frame_curr.idx))
        cv2.namedWindow("preview")
        cv2.imshow("preview", tracked_img[:, :, ::-1])
        cv2.waitKey(1)

    # # ! Print fieldnames
    # fieldnames = []
    # for term in terms:
    #     for ori in oris:
    #         for author in authors:
    #             fieldnames.append(author + "-" + term + "-" + ori)
    # write_report(os.path.join(output_dir, file), fieldnames)
    '''
    line = [
        np.quantile(error_n8p, 0.25, axis=0)[0],
        np.quantile(error_8p, 0.25, axis=0)[0],
        np.quantile(error_n8p, 0.25, axis=0)[1],
        np.quantile(error_8p, 0.25, axis=0)[1],
        # np.quantile(geo_error_n8p, 0.25, axis=0),
        # np.quantile(geo_error_8p, 0.25, axis=0),
        np.median(error_n8p, axis=0)[0],
        np.median(error_8p, axis=0)[0],
        np.median(error_n8p, axis=0)[1],
        np.median(error_8p, axis=0)[1],
        # np.median(geo_error_n8p, axis=0),
        # np.median(geo_error_8p, axis=0),
        np.quantile(error_n8p, 0.75, axis=0)[0],
        np.quantile(error_8p, 0.75, axis=0)[0],
        np.quantile(error_n8p, 0.75, axis=0)[1],
        np.quantile(error_8p, 0.75, axis=0)[1],
        # np.quantile(geo_error_n8p, 0.75, axis=0),
        # np.quantile(geo_error_8p, 0.75, axis=0),
        np.mean(error_n8p, axis=0)[0],
        np.mean(error_8p, axis=0)[0],
        np.mean(error_n8p, axis=0)[1],
        np.mean(error_8p, axis=0)[1],
        # np.mean(geo_error_n8p, axis=0),
        # np.mean(geo_error_8p, axis=0),
        np.std(error_n8p, axis=0)[0],
        np.std(error_8p, axis=0)[0],
        np.std(error_n8p, axis=0)[1],
        np.std(error_8p, axis=0)[1],
        # np.std(geo_error_n8p, axis=0),
        # np.std(geo_error_8p, axis=0)
    ]
    write_report(os.path.join(output_dir, file), line)
    '''

    print(
        "====================================================================="
    )
    # ! Ours' method
    print("Q1-ours:{} - {}".format(np.quantile(error_n8p, 0.25, axis=0),
                                   len(error_n8p)))
    print("Q2-ours:{} - {}".format(np.median(error_n8p, axis=0),
                                   len(error_n8p)))
    print("Q3-ours:{} - {}".format(np.quantile(error_n8p, 0.75, axis=0),
                                   len(error_n8p)))

    print(
        "====================================================================="
    )
    # ! 8PA
    print("Q1-8PA:{} -  {}".format(np.quantile(error_8p, 0.25, axis=0),
                                   len(error_8p)))
    print("Q2-8PA:{} -  {}".format(np.median(error_8p, axis=0), len(error_8p)))
    print("Q3-8PA:{} -  {}".format(np.quantile(error_8p, 0.75, axis=0),
                                   len(error_8p)))
    print(
        "====================================================================="
    )
