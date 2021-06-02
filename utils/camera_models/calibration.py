import os
from file_utilities import read_yaml_file
import numpy as np


class CalibrationBase:
    "Basic Calibration base class"

    def __init__(self, path):
        assert os.path.isfile(path), "file {} has not found".format(path)

    def get_intrinsics(self, **params):
        raise NotImplemented

    def get_distortion_coifs(self, **params):
        raise NotImplemented

    def get_resolution(self, **params):
        raise NotImplemented


class KalibrCalibration(CalibrationBase):
    """
    This class wraps the output information from a yaml calibration file from Kalibr
    more information about this format see. https://github.com/ethz-asl/kalibr/wiki/yaml-formats
    """

    def __init__(self, path):
        super(KalibrCalibration, self).__init__(path)
        self.camera_yaml = read_yaml_file(path)
        self.cameras_list = [cam for cam in self.camera_yaml]

    def get_extrinsics(self, idx):
        if idx == 0:
            return np.eye(4)
        cam = self.camera_yaml[self.cameras_list[idx]]
        extrinsic_list = cam["T_cn_cnm1"]
        extrinsic = np.eye(4)
        if "T_cn_cnm1" in cam.keys():
            for i, row in enumerate(extrinsic_list):
                extrinsic[i, :] = np.asarray(row)
        return extrinsic

    def get_intrinsics(self, idx):
        cam = self.camera_yaml[self.cameras_list[idx]]
        camera_model = cam["camera_model"]
        intrinsic_list = cam["intrinsics"]

        return camera_model, np.asarray(intrinsic_list)

    def get_distortion_coifs(self, idx):
        cam = self.camera_yaml[self.cameras_list[idx]]
        distortion_model = cam["distortion_model"]
        dist_coef_list = cam["distortion_coeffs"]

        return distortion_model, np.asarray(dist_coef_list)

    def get_resolution(self, idx):
        cam = self.camera_yaml[self.cameras_list[idx]]
        return cam["resolution"]


if __name__ == '__main__':
    _path = "/home/kike/Documents/PycharmProjects/StereoVision/StereoCalibration/stereo_calibration_ws/src/creating_rosbag/calibration_output/rosbag_2019-08-14-03-46-15.yaml"
    calibration = KalibrCalibration(_path)
    calibration.get_extrinsics(1)
    print(calibration.get_extrinsics(1))
