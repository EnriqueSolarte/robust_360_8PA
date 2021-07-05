import cv2
import numpy as np
from config import Cfg


class Shi_Tomasi_Extractor():
    def __init__(self, cfg: Cfg):
        self.extractor = "Shi_Tomasi"
        self.descriptor = cv2.ORB_create()
        self.feature_params = dict(maxCorners=cfg.params.max_number_corners,
                                   qualityLevel=cfg.params.quality_corner_level,
                                   minDistance=cfg.params.min_corner_distance,
                                   blockSize=cfg.params.block_size_for_corners)

    def get_features_descriptors(self, image, mask=None):
        if len(image.shape) > 2:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return self.descriptor.compute(image, self.get_features(image, mask))

    def get_features(self, image, mask=None):
        if len(image.shape) > 2:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        feature_points = cv2.goodFeaturesToTrack(image,
                                                 mask=mask,
                                                 **self.feature_params)
        return [
            cv2.KeyPoint(x=f[0][0], y=f[0][1], _size=20)
            for f in feature_points
        ]

    def get_tracked_image(self, image, mask=None):
        tracked_image = np.ones_like(image)
        kp, des = self.get_features_descriptors(image, mask)
        cv2.drawKeypoints(image, kp, tracked_image, color=(0, 255, 0), flags=0)
        return tracked_image, kp, des
