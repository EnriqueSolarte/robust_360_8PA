import os
import numpy as np
import cv2


class Data:
    def __init__(self, dt_dir, scene):
        self.dataset_name = None
        self.dt_dir = dt_dir
        self.scene = scene
        self.scene_dir = os.path.join(self.dt_dir, self.scene)
        assert os.path.isdir(self.scene_dir), "*** Scene directory does not exists {}".format(self.scene_dir)
        self.internal_index = 0
        self.camera_projection = None
        self.number_frames = 0
        self.shape = None

    def get_frame(self, idx, return_dict=False, return_depth=True):
        raise NotImplementedError

    def get_rgb(self, idx):
        raise NotImplementedError

    def get_pose(self, idx):
        raise NotImplementedError

    def get_shape(self):
        return self.shape

    def read(self):
        """
        This method coupling the function cv2.read() used to access to videos.
        It returns only images
        """
        if self.internal_index < self.number_frames:
            rgb, depth, pose, timestamps, idx = self.get_frame(self.internal_index)
            self.internal_index += 1
            return rgb, depth, pose, timestamps, idx, True
        else:
            return None, None, None, None, None, False
