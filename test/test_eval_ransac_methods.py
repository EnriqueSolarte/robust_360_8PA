from config import Cfg
from utils import *
from solvers import *


def eval_solvers(cfg: Cfg):
    sampler = BearingsSampler(cfg)

    list_methods = [
        get_cam_pose_by_ransac_8pa,
        get_cam_pose_by_ransac_opt_SK,
        get_cam_pose_by_ransac_GSM,
        get_cam_pose_by_ransac_GSM_const_wRT,
        get_cam_pose_by_ransac_GSM_const_wSK
    ]

    hist_errors = {}

    while True:
        bearings_kf, bearings_frm, cam_pose_gt, ret = sampler.get_bearings()
        if not ret:
            break
        
        if bearings_kf is not None:
            if bearings_kf.shape[1] < 8:
                continue
            
            print("#\n")
            print("Number of evaluated bearings: {}".format(bearings_kf.shape[1]))
            for method in list_methods:
                cam_pose_hat = method(
                    x1=bearings_kf,
                    x2=bearings_frm,
                    cfg=cfg
                )

                cam_pose_error = evaluate_error_in_transformation(
                    transform_est=cam_pose_hat,
                    transform_gt=cam_pose_gt
                )
                hist_errors = add_error_eval(cam_pose_error, method.__name__, hist_errors)
        
        print("done")


if __name__ == '__main__':

    config_file = Cfg.FILE_CONFIG_MP3D_VO
    # config_file = Cfg.FILE_CONFIG_TUM_VI

    cfg = Cfg.from_cfg_file(yaml_config=config_file)
    eval_solvers(cfg)
