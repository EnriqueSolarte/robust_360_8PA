from config import Cfg
import os
import save_bearings
import utils

if __name__ == '__main__':
    config_file = os.getenv("CFG_DEFAULT_FILE")
    cfg = Cfg.from_cfg_file(yaml_config=config_file)
    save_bearings.from_tracked_features(cfg)
    print("done")
    