from config import Cfg
import os
from imageio import imread
import matplotlib.pyplot as plt
import sys
sys.path.append("/home/kike/Documents/Research/Utilities")
sys.path.append("/home/kike/Documents/Research/CameraModels")

from TUM_RGBD_utils.evaluate_rpe import read_trajectory
from .dataset import Data
from projections.equi2pcl.equi2pcl import Equi2PCL
import numpy as np
from pcl_utilities import *


class MP3D_VO(Data):
    def __init__(self, dt_dir, scene):
        super().__init__(dt_dir, scene)
        self.dataset_name = "MP3D_VO"
        self.dir_rgb = self.scene_dir + "/rgb"
        self.dir_depth = self.scene_dir + "/depth/tiff"
        self.__ext_depth = "tiff"
        self.camera_poses = read_trajectory(self.scene_dir + "/frm_ref.txt")
        # self.camera_poses = pd.read_csv(self.dir_path + "/frm_ref.txt", header=None, sep=" ",).values
        self.number_frames = len(self.camera_poses)
        self.time_stamps = tuple(self.camera_poses.keys())
        if not self.number_frames == len(os.listdir(self.dir_rgb)):
            print(
                "Warn: Number of frames in frm_ref.txt do not match with rgb files"
            )
        color_map, depth_map, _, _ = self.get_frame(0)
        self.shape = (color_map.shape[0], color_map.shape[1])
        self.internal_index = 0
        # ! this allows us to project a equirectangular projection into pcl
        self.camera_projection = Equi2PCL(shape=self.shape)
        self.cam = self.camera_projection.cam

    def get_rgb(self, idx):
        try:
            rgb = imread(os.path.join(self.dir_rgb, "{}.png".format(idx + 1)))
        except:
            rgb = imread(os.path.join(self.dir_rgb, "{}.jpg".format(idx + 1)))

        return rgb
        
    def get_frame(self, idx, return_dict=False, **kwargs):
        assert idx < self.number_frames
        try:
            rgb = imread(os.path.join(self.dir_rgb, "{}.png".format(idx + 1)))
        except:
            rgb = imread(os.path.join(self.dir_rgb, "{}.jpg".format(idx + 1)))

        try:
            depth = imread(
                os.path.join(self.dir_depth,
                             "{}.{}".format(idx + 1, self.__ext_depth)))
        except:
            depth = None
            # print("No depth information !!!!")

        pose = self.camera_poses[self.time_stamps[idx]]
        rgb = np.array(rgb)

        if return_dict:
            return dict(image=rgb,
                        depth=depth,
                        pose=pose,
                        timestamp=self.time_stamps[idx],
                        idx=idx
                        )
        return rgb, depth, pose, self.time_stamps[idx], idx

    def get_shape(self):
        return self.shape

    def read(self):
        try:
            data = self.get_frame(self.internal_index, return_dict=True)
            self.internal_index += 1
        except:
            return None, None, None, False
        return data["image"],  self.internal_index-1, data["timestamp"], True

    def get_pose(self, idx):
        return self.camera_poses[self.time_stamps[idx]]


def get_default_mp3d_scene():
    scene = "1LXtFkjw3qL/0"
    path = "/home/kike/Documents/datasets/MP3D_VO"
    return MP3D_VO(dt_dir=path, scene=scene)


if __name__ == '__main__':
    basedir = "/home/kike/Documents/datasets/MP3D_VO"
    scene = "2n8kARJN3HM/0"

    assert os.path.isdir(basedir)
    dt = MP3D_VO(dt_dir=basedir, scene=scene)

    rgb, depth, pose, step = dt.get_frame(idx=150)
    plt.subplot(1, 2, 1)
    plt.imshow(rgb)
    plt.subplot(1, 2, 2)
    plt.imshow(depth)
    plt.show()
    import cv2
    cv2.imshow("test", rgb)
    cv2.waitKey(0)
    print("done")
