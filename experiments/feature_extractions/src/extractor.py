import cv2
import numpy as np


class ORBExtractor:
    def __init__(self,
                 nfeatures=int(200),
                 scaleFactor=1.5,
                 nlevels=2,
                 edgeThreshold=2,
                 patchSize=2):
        self.orb = cv2.ORB_create(
            nfeatures=nfeatures,
            scaleFactor=scaleFactor,
            nlevels=nlevels,
            edgeThreshold=edgeThreshold,
            patchSize=patchSize,
            scoreType=cv2.ORB_HARRIS_SCORE)

    def get_features_descriptors(self, image, mask=None):
        if len(image.shape) > 2:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return self.orb.compute(image, self.get_features(image, mask))

    def get_features(self, image, mask=None):
        if len(image.shape) > 2:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return self.orb.detect(image, mask)

    def get_tracked_image(self, image, mask=None):
        tracked_image = np.ones_like(image)
        kp, des = self.get_features_descriptors(image, mask)
        cv2.drawKeypoints(image, kp, tracked_image, color=(0, 255, 0), flags=0)
        return tracked_image, kp, des
