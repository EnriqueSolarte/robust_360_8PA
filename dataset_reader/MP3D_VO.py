import sys
import numpy as np
from utils.equi2pcl import Equi2PCL
from .dataset import Data
from utils.TUM_RGBD_utils.evaluate_rpe import read_trajectory
from config import Cfg
import os
from imageio import imread
import matplotlib.pyplot as plt



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
        self.timestamps = tuple(self.camera_poses.keys())
        if not self.number_frames == len(os.listdir(self.dir_rgb)):
            print(
                "Warn: Number of frames in frm_ref.txt do not match with rgb files"
            )
        color_map, depth_map, _, _, _ = self.get_frame(0)
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

        pose = self.camera_poses[self.timestamps[idx]]
        rgb = np.array(rgb)

        if return_dict:
            return dict(image=rgb,
                        depth=depth,
                        pose=pose,
                        timestamp=self.timestamps[idx],
                        idx=idx
                        )
        return rgb, depth, pose, self.timestamps[idx], idx

    def get_pose(self, idx):
        return self.camera_poses[self.timestamps[idx]]

    def get_pcl(self, idx):
        """
        By default, we expect that every dataset has depth information
        This method return a dense PCL by indexing the frame id
        """
        assert self.camera_projection is not None, "camera_projection is not defined"
        rgb, depth, pose, timestamps, idx = self.get_frame(
            idx, return_depth=True)

        if depth is None:
            print("Dataset {} does not have depth information".format(self.dataset_name))
            return None, None, pose, timestamps, idx
        pcl, color_pcl = self.camera_projection.get_pcl(
            color_map=rgb,
            depth_map=depth,
            scaler=self.camera_projection.scaler)
        return pcl, color_pcl, pose, timestamps, idx

    def get_pcl_from_key_features(self, idx, extractor, mask=None):
        """
        Returns a set of 3D-landmarks from a set of keyfeatures (keypoints)
        The projection of landmarks is based on the defined camera model and the
        depth map associated to the frame (color image) 
        """
        assert self.camera_projection is not None, "camera_projection is not defined"
        frame_dict = self.get_frame(idx, return_dict=True)
        # ! BY default any extractor return key-points (cv2.KeyPoints) from get_features()
        key_points = extractor.get_features(frame_dict["image"], mask=mask)
        pixels = np.array([kp.pt for kp in key_points]).astype(int)

        if frame_dict["depth"] is None:
            print("Dataset {} does not have depth information".format(self.dataset_name))
            return None

        depth = frame_dict["depth"][pixels[:, 1], pixels[:, 0]]
        msk_depth = depth > 0
        bearings = self.camera_projection.cam.pixel2euclidean_space(pixels)
        return bearings[:, msk_depth] * depth[msk_depth]
