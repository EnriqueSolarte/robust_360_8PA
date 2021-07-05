import sys
from .dataset import Data
import matplotlib.pyplot as plt
from utils.file_utilities import read_yaml_file
from utils.geometry_utilities import *
from utils.vispy_utilities import *
from utils.camera_models import Sphere
import cv2
import os
import numpy as np
from utils.TUM_RGBD_utils.associate import associate, read_file_list, read_file_association
from utils.TUM_RGBD_utils.evaluate_rpe import read_trajectory
from utils.TUM_RGBD_utils.save_association import save_association
from utils.camera_models.unifiedModel import UnifiedModel



class TUM_VI(Data):
    def __init__(self, dt_dir, scene):
        # Adding dataset utilities
        # print("loading TUM_RGBD_SLAM_data from: {}\nScene: {}".format(dataset_path, scene))
        # TUM_utilities = os.path.join(dataset_path, "utilities/scripts")
        # sys.path.append(TUM_utilities)
        super().__init__(dt_dir, scene)

        # from TUM_RGBD_utils.associate import associate, read_file_list, read_file_association
        # from TUM_RGBD_utils.evaluate_rpe import read_trajectory
        # from TUM_RGBD_utils.save_association import save_association
        # from unifiedModel import UnifiedModel

        rgb_image_paths = self.scene_dir + '/mav0/cam0/data.csv'
        trajectory_gt = self.scene_dir + '/dso/gt_imu.csv'

        # ! For now we are considering a monocular camera so
        # ! extrinsic parameters are neglected
        # camera_yaml = read_yaml_file(self.scene_dir + '/dso/camchain.yaml')
        camera_yaml = read_yaml_file(self.dt_dir + '/omni_calib.yaml')
        self.camera_projection = UnifiedModel.by_calibrated_parameters(calib=camera_yaml)

        # ! Create dic-bases
        self.dic_rgb = read_file_list(rgb_image_paths)
        self.dic_trajectory = read_file_list(trajectory_gt)
        self.dic_poses = read_trajectory(trajectory_gt, seq="wxyz")

        # ! data association
        pose_rgb_association_file = os.path.join(self.scene_dir, "dso/pose_rgb_association.txt")

        try:
            self.dic_pose_rgb_association = read_file_association(
                pose_rgb_association_file)
        except:
            self.dic_pose2image = dict(
                associate(self.dic_trajectory, self.dic_rgb, 0, 2 * 10 ** 9))
            data = dict(
                timestamp_pose=[time for time in self.dic_pose2image.keys()],
                timestamp_image=[self.dic_pose2image[key] for key in self.dic_pose2image.keys()]
            )
            save_association(
                caption="# Data association",
                file_association=pose_rgb_association_file,
                data=data
            )
            self.dic_pose_rgb_association = read_file_association(
                pose_rgb_association_file)

        self.number_frames = self.dic_pose_rgb_association.shape[0]
        self.timestamps = self.dic_pose_rgb_association[:, 0]
        self.t_cam_imu_transform = np.array(camera_yaml["cam0"]['T_cam_imu']).reshape((4, 4))
        self.shape = self.camera_projection.shape
        self.cam = self.camera_projection

    def get_shape(self):
        return self.camera_projection.get_shape()

    def get_rgb(self, idx):
        frame_dict = self.get_frame(idx, return_dict=True)
        return frame_dict["image"]

    def get_pose(self, idx):
        frame_dict = self.get_frame(idx, return_dict=True)
        return frame_dict["pose"]

    def get_frame(self, idx=0, return_dict=True):
        pose_timestamp = self.dic_pose_rgb_association[idx, 0]
        image_timestamp = self.dic_pose_rgb_association[idx, 1]

        image_file = self.scene_dir + '/mav0/cam0/data/' + self.dic_rgb[image_timestamp][0]
        assert os.path.isfile(image_file)

        rgb = cv2.imread(image_file)
        pose = np.array(self.dic_poses[pose_timestamp]) @ self.t_cam_imu_transform
        if return_dict:
            return dict(image=rgb,
                        depth=None,
                        pose=pose,
                        timestamp=self.timestamps[idx],
                        idx=idx
                        )

        return rgb, None, pose, self.timestamps[idx], idx
