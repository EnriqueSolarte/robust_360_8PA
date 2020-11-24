import os
from read_datasets.MP3D_VO import MP3D_VO
from analysis.utilities.data_utilities import track_features
from structures.tracker import LKTracker
from structures.extractor.shi_tomasi_extractor import Shi_Tomasi_Extractor
from solvers.epipolar_constraint import *
from analysis.utilities.plot_utilities import get_file_name
from file_utilities import create_dir, save_obj
from data_utilities import add_instance_to_dict
import pandas as pd
from analysis.utilities.data_utilities import sampling_bearings

solver = EightPointAlgorithmGeneralGeometry()


def save_data(**kwargs):
    dt = pd.DataFrame(np.vstack((kwargs["bearings"]["kf"], kwargs["bearings"]["frm"])).T)
    dirname = os.path.join(os.path.dirname(os.path.dirname(kwargs["filename"])),
                           "saved_bearings",
                           "saving_tracked_features" +
                           "_dist_" + str(kwargs["distance_threshold"]) +
                           "_res_" + str(kwargs["res"][0]) + "." + str(kwargs["res"][1]) +
                           "_loc_" + str(kwargs["loc"][0]) + "." + str(kwargs["loc"][1]) +
                           "_{}_".format(kwargs["extra"]),
                           "frames",
                           )

    kwargs["dirname"] = dirname
    file_bearings = str(kwargs["tracker"].initial_frame.idx) + "_" + str(
        kwargs["tracker"].tracked_frame.idx)
    file_bearings = os.path.join(dirname, file_bearings + ".txt")
    print("scene:{}".format(kwargs["data_scene"].scene))
    print("Frames Kf:{}-frm:{}".format(kwargs["tracker"].initial_frame.idx, kwargs["tracker"].tracked_frame.idx))
    print("tracked features {}".format(kwargs["bearings"]["kf"].shape[1]))
    create_dir(dirname, delete_previous=False)
    print(file_bearings)
    dt.to_csv(file_bearings, header=None, index=None)
    return kwargs


def saving_features(**kwargs):
    data_frame = dict()
    while True:
        bearings_kf, bearings_frm, cam_gt, kwargs, ret = track_features(**kwargs)
        if not ret:
            break

        kwargs["bearings"] = dict()
        kwargs["bearings"]["kf"] = bearings_kf
        kwargs["bearings"]["frm"] = bearings_frm
        # kwargs = sampling_bearings(**kwargs)

        cam_hat, e_hat, sigma, residuals = solver.evaluate_bearings(
            x1=kwargs["bearings"]["kf"],
            x2=kwargs["bearings"]["frm"]
        )

        errors = evaluate_error_in_transformation(
            transform_gt=cam_gt,
            transform_est=cam_hat
        )
        inliers_ratio = np.abs(residuals) < kwargs["threshold"]

        data_frame = add_instance_to_dict(label="rot_e", dict_=data_frame, xdata=errors[0])
        data_frame = add_instance_to_dict(label="tran_e", dict_=data_frame, xdata=errors[1])
        data_frame = add_instance_to_dict(label="sigma_8", dict_=data_frame, xdata=sigma[-2])
        data_frame = add_instance_to_dict(label="inliers", dict_=data_frame,
                                          xdata=np.sum(inliers_ratio) / len(inliers_ratio))

        kwargs["data_frame"] = data_frame
        kwargs = save_data(**kwargs)

    save_obj(os.path.join(os.path.dirname(kwargs["dirname"]), "frames_data.results"), kwargs["data_frame"])


if __name__ == '__main__':
    path = "/home/kike/Documents/datasets/MP3D_VO"
    scene = os.listdir(path)
    scene_list = os.listdir(path)
    label_info = "narrow_features"
    for sc in (scene_list[0],):
        scene = sc + "/0"
        data = MP3D_VO(scene=scene, basedir=path)

        settings = dict(
            data_scene=data,
            idx_frame=0,
            linear_motion=(-1, 1),
            angular_motion=(-10, 10),
            distance_threshold=0.5,
            res=(360, 180),
            loc=(0, 0),
            extra=label_info,
            skip_frames=100,
            noise=500,
            inliers_ratio=0.5,
            sampling=200,
            threshold=1e-4
        )
        features_setting = dict(
            feat_extractor=Shi_Tomasi_Extractor(maxCorners=225,
                                                qualityLevel=0.0001,
                                                minDistance=1,
                                                blockSize=5),
            tracker=LKTracker(lk_params=dict(winSize=(25, 25),
                                             maxLevel=4,
                                             criteria=(cv2.TERM_CRITERIA_EPS
                                                       | cv2.TERM_CRITERIA_COUNT, 10, 0.01))),
            show_tracked_features=True,
        )
        log_settings = dict(filename=get_file_name(file_src=__file__,
                                                   **settings, **features_setting,
                                                   create_directory=True))
        saving_features(**settings, **features_setting, **log_settings)
