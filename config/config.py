import cv2
import datetime
import yaml
import sys
import os
import csv
from collections import namedtuple


class Cfg:

    # ! Paths
    DIR_ROOT = os.getenv("DIR_ROOT")
    FILE_CONFIG_MP3D_VO = os.path.join(DIR_ROOT, 'config', 'config_MP3D_VO.yaml')
    FILE_CONFIG_TUM_VI = os.path.join(DIR_ROOT, 'config', 'config_TUM_VI.yaml')

    #! Datasets 
    DIR_MP3D_VO_DATASET = os.getenv("DIR_MP3D_VO_DATASET")
    DIR_TUM_VI_DATASET = os.getenv("DIR_TUM_VI_DATASET")

    # ! Labels
    #### ! camera poses
    CAM_POSES_GT = "cam_pose_gt.txt"
    CAM_POSES_8PA = "cam_pose_8PA.txt"
    CAM_POSES_eSK = "cam_pose_eSK.txt"
    CAM_POSES_GSM = "cam_pose_GSM.txt"
    CAM_POSES_wSK = "cam_pose_wSK.txt"

    #### ! camera poses
    FROM_TRACKED_BEARINGS= "tracked_bearings"
    FROM_SAMPLED_BEARINGS= "sampled_bearings"

    def __init__(self, *args, **kwargs):
        self.kwargs = kwargs
        self.prmt = namedtuple('ConfigFile', kwargs.keys())(*kwargs.values())
    
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

            # ! This is the comment at the end of every generated config file
            print('\n\n# VSLAB @ National Tsing Hua University')
            print("# This config file has been generated automatically")
            print("# for the bearing vectors saved at the following directory")
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
