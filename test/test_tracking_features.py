from config import Cfg
from utils import *

if __name__ == '__main__':

    config_file = Cfg.FILE_CONFIG_MP3D_VO
    config_file = Cfg.FILE_CONFIG_TUM_VI
    
    cfg = Cfg.from_cfg_file(yaml_config=config_file)
    tracker = FeatureTracker(cfg)

    while True:
        bearings_kf, bearings_frm, cam_pose_gt, ret = tracker.track()
        if not ret:
            break

