import os
from read_datasets.MP3D_VO import MP3D_VO
from read_datasets.TUM_VI import TUM_VI
from analysis.utilities.data_utilities import track_features
from structures.tracker import LKTracker
from structures.extractor.shi_tomasi_extractor import Shi_Tomasi_Extractor
import cv2
from solvers.epipolar_constraint import *
import numpy as np
import matplotlib.pyplot as plt
from data_utilities import *
from analysis.utilities.data_utilities import sampling_bearings
from suplementary.suppl_record_video_features import record_features
from suplementary.suppl_saving_features import saving_features
from analysis.utilities.plot_utilities import get_file_name

solver = EightPointAlgorithmGeneralGeometry()
threshold = 1e-4


def read_features(**kwargs):
    frames = []

    plt.show()
    plt.grid()
    axes = plt.gca()
    axes.set_xlim(0, 100)
    axes.set_ylim(-50, +50)
    line_1, = axes.plot([], [], 'r-', label="line 1")
    line_2, = axes.plot([], [], 'b-', label="line 2")
    line_1_list = []
    line_2_list = []
    line_1_med_list = []
    line_2_med_list = []
    i = 0

    while True:
        bearings_kf, bearings_frm, cam_gt, kwargs, ret = track_features(**kwargs)
        if not ret:
            break

        kwargs["bearings"] = dict()
        kwargs["bearings"]["kf"] = bearings_kf
        kwargs["bearings"]["frm"] = bearings_frm

        cam_hat, e_hat, sigma, residuals = solver.evaluate_bearings(
            x1=kwargs["bearings"]["kf"],
            x2=kwargs["bearings"]["frm"]
        )

        errors = evaluate_error_in_transformation(
            transform_gt=cam_gt,
            transform_est=cam_hat
        )

        inliers_ratio = np.abs(residuals) < kwargs["threshold"]

        line_1_list.append(sigma[-2])
        line_2_list.append(np.sum(inliers_ratio) / len(inliers_ratio))
        #
        # line_1_list.append(errors[0])
        # line_2_list.append(errors[1])
        #
        line_1_med_list.append(sigma[-2])
        line_2_med_list.append(np.median(line_2_list))

        max_value = np.max((np.max(line_1_med_list), np.max(line_2_med_list)))

        frames.append(i)
        axes.set_xlim(0, i + 10)
        axes.set_ylim(0, max_value * 1.1)
        line_1.set_xdata(frames)
        line_2.set_xdata(frames)

        line_1.set_ydata(line_1_med_list)
        line_2.set_ydata(line_2_med_list)

        plt.draw()
        plt.legend()
        plt.pause(1e-17)
        i += 1

    plt.show()


if __name__ == '__main__':
    # path = "/home/kike/Documents/datasets/MP3D_VO"
    # scene = os.listdir(path)
    # scene_list = os.listdir(path)
    path = "/home/kike/Documents/datasets/TUM_VI/dataset"
    scene_list = os.listdir(path)
    #
    # # scene = "dataset-room3_512_16"
    # scene = "dataset-room1_512_16"

    for sc in scene_list:
        scene = sc + "/1"
        # data = MP3D_VO(scene=scene, basedir=path)
        data = TUM_VI(scene=sc, basedir=path)

        settings = dict(
            data_scene=data,
            idx_frame=0,
            linear_motion=(-1, 1),
            angular_motion=(-10, 10),
            distance_threshold=0.5,
            res=(360, 180),
            loc=(0, 0),
            skip_frames=100,
            noise=500,
            inliers_ratio=0.5,
            sampling=200,
            threshold=8e-5
        )
        features_setting = dict(
            feat_extractor=Shi_Tomasi_Extractor(maxCorners=220,
                                                qualityLevel=0.0001,
                                                minDistance=20,
                                                blockSize=1),
            # extra="normal_scene",
            extra="spinning_scene",

            tracker=LKTracker(lk_params=dict(winSize=(25, 25),
                                             maxLevel=4,
                                             criteria=(cv2.TERM_CRITERIA_EPS
                                                       | cv2.TERM_CRITERIA_COUNT, 2, 0.01))),
            show_tracked_features=True,
        )
        log_settings = dict(filename=get_file_name(file_src=__file__,
                                                   **settings, **features_setting,
                                                   create_directory=True))
        # read_features(**settings, **features_setting)
        record_features(**settings, **features_setting)
        # saving_features(**settings, **features_setting, **log_settings)
