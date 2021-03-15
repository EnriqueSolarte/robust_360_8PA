import os
import numpy as np
import cv2


class Data:
    def __init__(self, dt_dir, scene):
        self.dt_dir = dt_dir
        self.scene = scene
        self.scene_dir = os.path.join(self.dt_dir, self.scene)
        assert os.path.isdir(self.scene_dir), "*** Scene directory does not exists {}".format(self.scene_dir)
        self.internal_index = 0
        self.camera_projection = None
        self.number_frames = 0

    def get_frame(self, idx, return_dict=False, return_depth=True):
        raise NotImplementedError

    def get_shape(self):
        raise NotImplementedError

    def read(self):
        """
        This method coupling the function cv2.read() used to access to videos.
        It returns only images
        """
        if self.internal_index < self.number_frames:
            dic_dt = self.get_frame(self.internal_index, return_dict=True)
            self.internal_index += 1
            return True, dic_dt["image"]
        else:
            return False, None

    def get_pcl(self, idx):
        """
        By default, we expect that every dataset has depth information
        This method return a dense PCL by indexing the frame id
        """
        assert self.camera_projection is not None, "camera_projection is not defined"
        color_map, depth_map, pose_, timestamp_ = self.get_frame(
            idx, return_depth=True)
        pcl, color_pcl = self.camera_projection.get_pcl(
            color_map=color_map,
            depth_map=depth_map,
            scaler=self.camera_projection.scaler)
        return pcl, color_pcl, pose_, timestamp_

    def get_pcl_from_key_features(self, idx, extractor, mask=None):
        assert self.camera_projection is not None, "camera_projection is not defined"
        frame_dict = self.get_frame(idx, return_dict=True)
        # ! BY default any extractor return key-points (cv2.KeyPoints) from get_features()
        key_points = extractor.get_features(frame_dict["image"], mask=mask)
        pixels = np.array([kp.pt for kp in key_points]).astype(int)
        depth = frame_dict["depth"][pixels[:, 1], pixels[:, 0]]
        msk_depth = depth > 0
        bearings = self.camera_projection.cam.pixel2euclidean_space(pixels)
        return bearings[:, msk_depth] * depth[msk_depth]

    def print_projection_plt(self, sparse_pcl, image, size=5):
        """ projects pcl points into camera image """
        mask = sparse_pcl[2, :] > 0
        z = sparse_pcl[2, mask]
        hm = (sparse_pcl[0:3, mask] / z)
        px = np.round(self.camera_projection.K @ hm) - 1
        hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        for i in range(px.shape[1]):
            # print("pixel", (px[1][i], px[0][i]))
            cv2.circle(hsv_image, (np.int32(px[0][i]), np.int32(px[1][i])),
                       size, (int(z[i]), 255, 255), -1)

        return cv2.cvtColor(hsv_image, cv2.COLOR_HSV2RGB)
