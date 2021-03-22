import cv2
import datetime
import yaml
import sys
import os
import csv


class Cfg:
    DIR_ROOT = os.getenv("DIR_ROOT")
    FILE_CONFIG_MP3D_VO = os.path.join(DIR_ROOT, 'config', 'config_MP3D_VO.yaml')
    FILE_CONFIG_TUM_VI = os.path.join(DIR_ROOT, 'config', 'config_TUM_VI.yaml')

    DIR_MP3D_VO_DATASET = os.getenv("DIR_MP3D_VO_DATASET")
    DIR_TUM_VI_DATASET = os.getenv("DIR_TUM_VI_DATASET")

    def __init__(self, *args, **kwargs):
        self.kwargs = kwargs

        try:
            # ! Data settings
            self.scene = kwargs["scene"]
            self.scene_version = str(kwargs["scene_version"])
            self.dataset_name = kwargs["dataset"]

            # ! Tracking features settings
            self.initial_frame = kwargs["initial_frame"]
            self.special_tracking = kwargs["special_tracking"]
            self.min_cam_distance = kwargs["min_cam_distance"]
            self.show_tracked_features = kwargs["show_tracked_features"]
            self.save_bearings = kwargs["save_bearings"]

            # ! Shi-Tomasi Feature extrator
            self.max_number_corners = kwargs["max_number_corners"]
            self.quality_corner_level = kwargs["quality_corner_level"]
            self.min_corner_distance = kwargs["min_corner_distance"]
            self.block_size_for_corners = kwargs["block_size_for_corners"]

            # ! LK tracker
            self.coarse_fine_levels = kwargs["coarse_fine_levels"]
            self.block_size_for_tracking = kwargs["block_size_for_tracking"]
            self.eps_tracking = kwargs["eps_tracking"]
            self.counter_iterations = kwargs["counter_iterations"]
        except:
            print("Error reading YAML config file")

    def save_config(self, dirname):
        """
        Saves the current configuration (settings) in a yaml file
        """
        time = datetime.datetime.now()

        config_file = os.path.join(dirname, "config.yaml")

        timestamp = str(time.year) + "-" + str(time.month) + "-" + str(time.day) + \
            "." + str(time.hour) + '.' + str(time.minute) + '.' + str(time.second)
        original_stdout = sys.stdout  # Save a reference to the original standard output

        with open(config_file, "w") as file:
            yaml.dump(self.kwargs, file)

            sys.stdout = file  # Change the standard output to the file we created.

            print('\n\n# VSLAB National Tsing Hua University')
            print("# This config file has been generated automatically with the parameters")
            print("# used for tracking the features described in this directory")
            print("# {}".format(dirname))
            print('# {}'.format(timestamp))
            sys.stdout = original_stdout

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
