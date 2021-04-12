from config import Cfg
from utils import *
from solvers import G8PA

if __name__ == '__main__':

    config_file = Cfg.FILE_CONFIG_MP3D_VO
    # config_file = Cfg.FILE_CONFIG_TUM_VI

    cfg = Cfg.from_cfg_file(yaml_config=config_file)
    tracker = FeatureTracker(cfg)
    g8p = G8PA(cfg)

    while True:
        bearings_kf, bearings_frm, cam_pose_gt, ret = tracker.track(verbose=False)
        if not ret:
            break

        if bearings_kf is not None:
            cam_pose_8pa = g8p.recover_pose_from_matches(
                x1=bearings_kf,
                x2=bearings_frm
            )

            error_8pa = evaluate_error_in_transformation(
                transform_est=cam_pose_8pa,
                transform_gt=cam_pose_gt
                )

            print("####################")
            print("Error in camera pose")
            print("Rot-e:{}\n Trans-e:{}".format(error_8pa[0], error_8pa[1]))
