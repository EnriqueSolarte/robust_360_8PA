############## Setting paths ################
import cv2
import datetime
import yaml
import sys
import os
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())
sys.path.append(os.getenv("MAIN_DIR"))
#################******#####################


class Cfg:
    def __init__(self, *args, **kwargs):
        self.kwargs = kwargs

        # ! Data settings
        self.scene = kwargs.get("scene", None)
        self.scene_version = str(kwargs.get("scene_version", 0))
        self.dataset = kwargs["dataset"]

        # ! Tracking features settings
        self.initial_frame = kwargs.get("initial_frame", 0)
        self.distance_threshold = kwargs.get("distance_threshold", 0.5)
        self.fov = kwargs.get("fov", (360, 180))
        self.loc = kwargs.get("loc", (0, 0))
        self.show_tracked_features = kwargs.get("show_tracked_features", True)
        self.save_bearings = kwargs.get("save_bearings", True)
        
        # ! Shi-Tomasi Feature extrator
        self.maxCorners= kwargs.get("max_num_corners", 500)
        self.qualityLevel= kwargs.get("qlt_level", 0.0001)
        self.minDistance= kwargs.get("min_distance_corners", 1)
        self.blockSize= kwargs.get("block_size", 10)

        # ! LK tracker
        self.coarseLevels = kwargs.get("max_levels", 2)
        
        

    
    def save_config(self):
        """
        Saves the current configuration (settings) in a yaml file
        """
        time = datetime.datetime.now()
        timestamp = str(time.year) + "-" + str(time.month) + "-" + str(time.day) + \
            "." + str(time.hour) + '.' + str(time.minute) + '.' + str(time.second)

        config_file = os.path.join(os.getenv("CONF_DIR"), "{}.yaml".format(timestamp))

        with open(config_file, "w") as file:
            yaml.dump(self.kwargs, file)

    @staticmethod
    def get_feature_extrator(*args, **kwargs):
        """
        Returns an instance of the features extractor Shi Tomasi
        """
        feature_extractor = Shi_Tomasi_Extractor(
            maxCorners=500,
            qualityLevel=0.001,
            minDistance=10,
            blockSize=10,
            descriptor_method=cv2.ORB_create()
        )
        return feature_extractor

    def get_lk_tracker(self, *args, **kwargs):
        tracker = LKTracker(
            lk_params=dict(winSize=(10, 10),
                           maxLevel=4,
                           criteria=(
                cv2.TERM_CRITERIA_EPS
                | cv2.TERM_CRITERIA_COUNT, 5, 0.0001)))

        tracker.extractor = self.get_feature_extrator(**kwargs)
        return tracker

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
