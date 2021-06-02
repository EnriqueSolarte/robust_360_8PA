from config import Cfg
from utils import *

if __name__ == '__main__':

    config_file = Cfg.FILE_CONFIG_MP3D_VO
    cfg = Cfg.from_cfg_file(yaml_config=config_file)

    saved_data = SavedData(cfg=cfg, tracked_or_sampled=cfg.FROM_TRACKED_BEARINGS)

    while True:
        data_bearings, ret = saved_data.get_bearings(return_dict=True)
        bearings_kf, bearings_frm, cam_pose_gt, ret = saved_data.get_bearings()
        if not ret:
            break

        print("reading {} ".format(saved_data.list_bearings_files[saved_data.idx-1]))