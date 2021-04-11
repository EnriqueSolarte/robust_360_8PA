from config import Cfg
from utils import *


if __name__ == '__main__':

    config_file = Cfg.FILE_CONFIG_MP3D_VO
    cfg = Cfg.from_cfg_file(yaml_config=config_file)

    sampler = BearingsSampler(cfg)

    while True:
        data_bearings, ret = sampler.get_bearings(return_dict=True)
        if not ret:
            break

        save_bearings(**data_bearings, save_config=True, save_camera_as="cam_gt.txt")
