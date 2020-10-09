from solvers.epipolar_constraint import EightPointAlgorithmGeneralGeometry as g8p
from solvers.optimal8pa import Optimal8PA as g8p_norm
from read_datasets.MP3D_VO import MP3D_VO
from structures.extractor.orb_extractor import ORBExtractor
from structures.extractor.shi_tomasi_extractor import Shi_Tomasi_Extractor
from structures.tracker import LKTracker
from structures.frame import Frame
from geometry_utilities import *
import cv2
from image_utilities import get_mask_map_by_res_loc

error_8p = []
error_n8p = []


def eval_camera_pose(**kwargs):
    g8p_n = g8p_norm(version=kwargs["opt_version"])
    g8p_ = g8p()
    cam_gt = kwargs["cam_gt"]
    cam = Sphere(shape=kwargs["tracker"].initial_frame.shape)
    matches = kwargs["tracker"].get_matches()
    bearings_kf = cam.pixel2euclidean_space(matches[0])
    bearings_frm = cam.pixel2euclidean_space(matches[1])

    cam_8p = g8p_.recover_pose_from_matches(x1=bearings_kf.copy(),
                                            x2=bearings_frm.copy())

    cam_8p_norm = g8p_n.recover_pose_from_matches(x1=bearings_kf.copy(),
                                                  x2=bearings_frm.copy())

    error_8p.append(
        evaluate_error_in_transformation(transform_gt=cam_gt,
                                         transform_est=cam_8p))
    error_n8p.append(
        evaluate_error_in_transformation(transform_gt=cam_gt,
                                         transform_est=cam_8p_norm))

    print(
        "====================================================================="
    )
    # ! Ours' method
    print("Q1-ours:{} -{}".format(np.quantile(error_n8p, 0.25, axis=0),
                                  len(error_n8p)))
    print("Q2-ours:{} -{}".format(np.median(error_n8p, axis=0),
                                  len(error_n8p)))
    print("Q3-ours:{} -{}".format(np.quantile(error_n8p, 0.75, axis=0),
                                  len(error_n8p)))

    print(
        "====================================================================="
    )
    # ! 8PA
    print("Q1-8PA:{} - {}".format(np.quantile(error_8p, 0.25, axis=0),
                                  len(error_8p)))
    print("Q2-8PA:{} - {}".format(np.median(error_8p, axis=0), len(error_8p)))
    print("Q3-8PA:{} - {}".format(np.quantile(error_8p, 0.75, axis=0),
                                  len(error_8p)))
    print(
        "====================================================================="
    )


def run_sequence_of_frames(**kwargs):
    if 'scene' in kwargs.keys():
        scene = kwargs["scene"]
        path = kwargs["basedir"]
        kwargs["data_scene"] = MP3D_VO(scene=scene, basedir=path)

    kwargs["mask"] = get_mask_map_by_res_loc(kwargs["data_scene"].shape,
                                             res=kwargs["res"],
                                             loc=kwargs["loc"])
    initial_frame = kwargs["idx_frame"]
    idx = initial_frame
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
        idx += 1
        relative_pose = frame_curr.get_relative_pose(
            key_frame=kwargs["tracker"].initial_frame)
        camera_distance = np.linalg.norm(relative_pose[0:3, 3])

        tracked_img = kwargs["tracker"].track(frame=frame_curr)

        print("Camera Distance       {}".format(camera_distance))
        print("Tracked features      {}".format(len(kwargs["tracker"].tracks)))
        print("KeyFrame/CurrFrame:   {}-{}".format(
            kwargs["tracker"].initial_frame.idx, frame_curr.idx))
        cv2.imshow("preview", tracked_img[:, :, ::-1])
        cv2.waitKey(10)

        if camera_distance > kwargs["distance_threshold"]:
            kwargs["cam_gt"] = relative_pose
            eval_camera_pose(**kwargs)
            break

    print("done")


if __name__ == '__main__':
    path = "/home/kike/Documents/datasets/MP3D_VO"
    scene = "2azQ1b91cZZ/0"
    # scene = "1LXtFkjw3qL/0"
    # scene = "759xd9YjKW5/0"
    # path = "/run/user/1001/gvfs/sftp:host=140.114.27.95,port=50002/NFS/kike/minos/vslab_MP3D_VO/512x1024"
    data = MP3D_VO(scene=scene, basedir=path)

    scene_settings = dict(
        data_scene=data,
        idx_frame=549,
        distance_threshold=0.5,
        res=(65.5, 46.4),
        # res=(180, 180),
        loc=(0, 0),
        # feat_extractor=ORBExtractor(),
        feat_extractor=Shi_Tomasi_Extractor(),
        tracker=LKTracker(),
        opt_version="v1.1",
    )
    run_sequence_of_frames(**scene_settings)
