import cv2
import datetime
import yaml
import sys
import os


class Cfg:
    DIR_ROOT = os.getenv("DIR_ROOT")
    FILE_CONFIG_MP3D_VO = os.path.join(DIR_ROOT, 'config', 'config_MP3D_VO.yaml')
    FILE_CONFIG_TUM_VI = os.path.join(DIR_ROOT, 'config', 'config_TUM_VI.yaml')

    DIR_MP3D_VO_DATASET = os.getenv("DIR_MP3D_VO_DATASET")
    DIR_TUM_VI_DATASET = os.getenv("DIR_TUM_VI_DATASET")

    def __init__(self, *args, **kwargs):
        self.kwargs = kwargs

        # ! Data settings
        self.scene = kwargs.get("scene")
        self.scene_version = str(kwargs.get("scene_version"))
        self.dataset_name = kwargs["dataset"]

        # ! Tracking features settings
        self.initial_frame = kwargs.get("initial_frame")
        self.special_tracking = kwargs.get("special_tracking")
        self.min_cam_distance = kwargs.get("min_cam_distance")
        self.show_tracked_features = kwargs.get("show_tracked_features")
        self.save_bearings = kwargs.get("save_bearings")

        # ! Shi-Tomasi Feature extrator
        self.max_number_corners = kwargs.get("max_number_corners")
        self.quality_corner_level = kwargs.get("quality_corner_level")
        self.min_corner_distance = kwargs.get("min_corner_distance")
        self.block_size_for_corners = kwargs.get("block_size_for_corners")

        # ! LK tracker
        self.coarse_fine_levels = kwargs.get("coarse_fine_levels")
        self.block_size_for_tracking = kwargs.get("block_size_for_tracking")
        self.eps_tracking = kwargs.get("eps_tracking")
        self.counter_iterations = kwargs.get("counter_iterations")

    @classmethod
    def from_cfg_file(cls, yaml_config):
        """
        Loads a set configuration parameters described by a yaml file
        """
        assert os.path.isfile(yaml_config), "Configuration yaml does not exit: {}".format(yaml_config)
        with open(yaml_config) as file:
            try:
                cfg_dict = yaml.load(file, Loader=yaml.SafeLoader)
            except yaml.YAMLError as e:
                print(e)

        cfg = cls(**cfg_dict)
        return cfg


if __name__ == '__main__':
    file = "/home/kike/Documents/Research/robust_360_8PA/config/experimets_1.yaml"
    cfg = Cfg.from_cfg_file(yaml_config=file)
