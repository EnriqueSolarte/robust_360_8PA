from file_utilities import generate_fingerprint_time
import os
from read_datasets.MP3D_VO import MP3D_VO
from analysis.utilities.data_utilities import get_bearings_by_plc
from file_utilities import create_dir
import pandas as pd
import numpy as np
from file_utilities import save_obj


def save_bearings_from_sampled_pcl(**kwargs):
    scene_directory = os.path.join(
        os.path.dirname(__file__), "results",
        kwargs["data_scene"].scene,
        os.path.splitext(os.path.split(__file__)[1])[0] + "_" +
        kwargs["extra"] +
        "_samples_" + str(kwargs["sampling"]) +
        "_inliers_" + str(kwargs["inliers_ratio"]) +
        "_noise_" + str(kwargs["noise"]) +
        "_res_" + str(kwargs["res"][0]) + "." + str(kwargs["res"][1]) +
        "_loc_" + str(kwargs["loc"][0]) + "." + str(kwargs["loc"][1])

    )
    create_dir(scene_directory, delete_previous=True)

    # ! Save settings
    dt = {d: kwargs[d] for d in kwargs.keys() if d not in "data_scene"}
    filename = os.path.join(scene_directory, "settings.config")
    save_obj(filename, dt)

    list_poses = []
    while True:
        kwargs, ret = get_bearings_by_plc(**kwargs)
        if not ret:
            break
        # ! Save current generated bearings
        dt = pd.DataFrame(np.vstack((kwargs["bearings"]["kf"], kwargs["bearings"]["frm"])).T)
        filename = os.path.join(scene_directory, str(kwargs["results"]["kf"][-1]) + ".txt")
        dt.to_csv(filename, header=None, index=None)
        print("saved file: {}".format(filename))
        list_poses.append(kwargs["cam_gt"].flatten())

    # ! Save all camera poses
    dt = pd.DataFrame(list_poses)
    filename = os.path.join(scene_directory, "cam_poses.txt")
    dt.to_csv(filename, header=None, index=None)


if __name__ == '__main__':
    path = "/home/kike/Documents/datasets/MP3D_VO"
    scene_list = os.listdir(path)
    label_info = generate_fingerprint_time()
    for sc in scene_list:
        scene = sc + "/0"
        data = MP3D_VO(scene=scene, basedir=path)

        settings = dict(
            data_scene=data,
            idx_frame=0,
            linear_motion=(-1, 1),
            angular_motion=(-10, 10),
            res=(360, 180),
            loc=(0, 0),
            extra=label_info,
            skip_frames=1,
            noise=500,
            inliers_ratio=0.9,
            sampling=200,
        )
        save_bearings_from_sampled_pcl(**settings)
