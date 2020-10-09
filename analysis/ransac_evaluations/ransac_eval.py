from read_datasets.MP3D_VO import MP3D_VO
from analysis.utilities.data_utilities import get_bearings
from solvers.ransac.ransac_8pa import RANSAC_8PA
from solvers.ransac.ransac_opt_8pa import RANSAC_OPT_8PA
from geometry_utilities import evaluate_error_in_transformation
import numpy as np
from structures.tracker import LKTracker
from structures.extractor.shi_tomasi_extractor import Shi_Tomasi_Extractor


def eval_methods(**kwargs):
    rot_error_8pa, tran_error_8pa, time_8pa = [], [], []
    rot_error_ours, tran_error_ours, time_ours = [], [], []

    while True:
        kwargs, ret = get_bearings(**kwargs)
        if not ret:
            break
        bearings_kf = kwargs["bearings"]["kf"]
        bearings_frm = kwargs["bearings"]["frm"]
        cam_gt = kwargs["cam_gt"]

        ransac = RANSAC_8PA(**kwargs)
        cam_pose = ransac.get_cam_pose(
            bearings_1=bearings_kf,
            bearings_2=bearings_frm,
        )

        error = evaluate_error_in_transformation(
            transform_est=cam_pose,
            transform_gt=cam_gt
        )
        rot_error_8pa.append(error[0])
        tran_error_8pa.append(error[1])
        time_8pa.append(ransac.time_evaluation)

        print("=================================================")
        print("8PA: rot_error {}".format(np.median(rot_error_8pa)))
        print("8PA: tran_error {}".format(np.median(tran_error_8pa)))
        print("inliers:{}".format(ransac.best_inliers_num))
        print("trials:{}".format(ransac.counter_trials))
        print("best_evaluation:{}".format(ransac.best_evaluation))
        print("time_evaluation:{}".format(np.median(time_8pa)))
        print("done")

        ransac = RANSAC_OPT_8PA(**kwargs)
        cam_pose = ransac.get_cam_pose(
            bearings_1=bearings_kf,
            bearings_2=bearings_frm,
        )

        error = evaluate_error_in_transformation(
            transform_est=cam_pose,
            transform_gt=cam_gt
        )
        rot_error_ours.append(error[0])
        tran_error_ours.append(error[1])
        time_ours.append(ransac.time_evaluation)
        print("=================================================")
        print("OURS: rot_error {}".format(np.median(rot_error_ours)))
        print("OURS: tran_error {}".format(np.median(tran_error_ours)))
        print("inliers:{}".format(ransac.best_inliers_num))
        print("trials:{}".format(ransac.counter_trials))
        print("best_evaluation:{}".format(ransac.best_evaluation))
        print("time_evaluation:{}".format(np.median(time_ours)))


if __name__ == '__main__':
    path = "/home/kike/Documents/datasets/MP3D_VO"
    scene = "2azQ1b91cZZ/0"
    # scene = "i5noydFURQK/0"
    # scene = "sT4fr6TAbpF/0"
    # scene = "1LXtFkjw3qL/0"
    # scene = "759xd9YjKW5/0"

    data = MP3D_VO(scene=scene, basedir=path)
    scene_settings = dict(
        data_scene=data,
        idx_frame=0,
        linear_motion=(-1, 1),
        angular_motion=(-10, 10),
        res=(360, 180),
        loc=(0, 0),
        sampling=200,
        noise=500,
        inliers_ratio=0.5
    )
    features_setting = dict(
        feat_extractor=Shi_Tomasi_Extractor(maxCorners=1000),
        tracker=LKTracker(),
        show_tracked_features=False,
        special_eval=False)

    ransac_settings = dict(
        expected_inliers=0.5,
        probability_success=0.99,
        residual_threshold=1e-3,
    )

    eval_methods(**scene_settings, **ransac_settings, **features_setting)
