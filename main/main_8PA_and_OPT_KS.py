from config import Cfg
from utils import *
from solvers import get_cam_pose_by_8pa, get_cam_pose_by_opt_SK


def eval_solvers(cfg: Cfg):
    
    # tracker = FeatureTracker(cfg)
    sampler = BearingsSampler(cfg)

    rot_e_list_8PA = []
    tra_e_list_8PA = []

    rot_e_list_opt_SK = []
    tra_e_list_opt_SK = []

    while True:
        # bearings_kf, bearings_frm, cam_pose_gt, ret = tracker.track(verbose=False)
        bearings_kf, bearings_frm, cam_pose_gt, ret = sampler.get_bearings()
        if not ret:
            break

        if bearings_kf is not None:

            cam_pose_opt_SK = get_cam_pose_by_opt_SK(
                x1=bearings_kf,
                x2=bearings_frm
            )

            cam_pose_8pa = get_cam_pose_by_8pa(
                x1=bearings_kf,
                x2=bearings_frm
            )

            error_8pa = evaluate_error_in_transformation(
                transform_est=cam_pose_8pa,
                transform_gt=cam_pose_gt
            )

            error_opt_SK = evaluate_error_in_transformation(
                transform_est=cam_pose_opt_SK,
                transform_gt=cam_pose_gt
            )

            rot_e_list_8PA.append(error_8pa[0])
            tra_e_list_8PA.append(error_8pa[1])

            rot_e_list_opt_SK.append(error_opt_SK[0])
            tra_e_list_opt_SK.append(error_opt_SK[1])

            print("####################")
            print("Error in camera pose")
            print("8PA\tRot-e:{}\tTra-e:{}".format(
                np.median(rot_e_list_8PA),
                np.median(tra_e_list_8PA))
            )

            print("Opt SK\tRot-e:{}\tTra-e:{}".format(
                np.median(rot_e_list_opt_SK),
                np.median(tra_e_list_opt_SK))
            )
            print("Number of bearings: {}".format(bearings_kf.shape[1]))


if __name__ == '__main__':

    config_file = Cfg.FILE_CONFIG_MP3D_VO
    # config_file = Cfg.FILE_CONFIG_TUM_VI

    cfg = Cfg.from_cfg_file(yaml_config=config_file)
    eval_solvers(cfg)
