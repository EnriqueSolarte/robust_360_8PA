from config import Cfg
from tracker import FeatureTracker
from utils import save_bearings

if __name__ == '__main__':

    config_file = Cfg.FILE_CONFIG_MP3D_VO
    config_file = Cfg.FILE_CONFIG_TUM_VI

    cfg = Cfg.from_cfg_file(yaml_config=config_file)
    tracker = FeatureTracker(cfg)

    while True:
        data_bearings, ret = tracker.track(return_dict=True)
        if not ret:
            break

        save_bearings(**data_bearings, save_config=True)