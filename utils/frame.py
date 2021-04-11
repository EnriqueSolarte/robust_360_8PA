import cv2
import numpy as np


class Frame:
    def __init__(self, image, depth, pose, timestamp, idx):
        self.timestamp = timestamp
        self.idx = idx
        self.shape = image.shape[0], image.shape[1]
        self.depth = depth
        self.color_map = image
        if len(image.shape) > 2:
            self.grey_map = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            self.grey_map = image
        self.pose_wc = pose
        self.key_points, self.descriptors = None, None

    def define_features(self, extractor, mask=None):
        self.key_points, self.descriptors = extractor.get_features_descriptors(
            image=self.grey_map, mask=mask)

    def get_relative_pose(self, key_frame):
        """
        Returns the camera pose from the passed key_frame to the current frame (THIS)
        """
        return np.linalg.inv(key_frame.pose_wc).dot(self.pose_wc)
