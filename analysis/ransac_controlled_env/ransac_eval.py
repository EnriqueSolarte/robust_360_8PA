from read_datasets.MP3D_VO import MP3D_VO
from analysis.utilities.data_utilities import get_bearings_by_plc
from solvers.ransac.ransac_8pa import RANSAC_8PA
from geometry_utilities import evaluate_error_in_transformation


def eval_methods(**kwargs):
    while True:
        bearings_kf, bearings_frm, cam_gt, kwargs = get_bearings_by_plc(**kwargs)
        ransac = RANSAC_8PA(**kwargs)
        cam_pose = ransac.get_cam_pose(
            bearings_1=bearings_kf,
            bearings_2=bearings_frm,
        )

        error = evaluate_error_in_transformation(
            transform_est=cam_pose,
            transform_gt=cam_gt
        )
        print("=================================================")
        print("inliers:{}".format(ransac.best_inliers_num))
        print("trials:{}".format(ransac.counter_trials))
        print("best_evaluation:{}".format(ransac.best_evaluation))
        print("time_evaluation:{}".format(ransac.time_evaluation))
        print(error)
        print("done")


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

    ransac_settings = dict(
        expected_inliers=0.5,
        probability_success=0.99,
        residual_threshold=1.3e-4
    )

    eval_methods(**scene_settings, **ransac_settings)
