from read_datasets.MP3D_VO import MP3D_VO
from structures.extractor.shi_tomasi_extractor import Shi_Tomasi_Extractor
from structures.tracker import LKTracker
from analysis.utilities.data_utilities import track_features
from solvers.ransac.ransac_ours_8pa import RANSAC_OURS_8PA
from geometry_utilities import evaluate_error_in_transformation


def run_example(**kwargs):
    bearings_kf, bearings_frm, cam_gt, kwargs, ret = track_features(**kwargs)
    if not ret:
        exit(1)

    cam_pose = RANSAC_OURS_8PA().get_cam_pose(
        bearings_1=bearings_kf,
        bearings_2=bearings_frm,
    )
    error = evaluate_error_in_transformation(
        transform_est=cam_pose,
        transform_gt=cam_gt
    )
    print(error)


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
        distance_threshold=0.5,
        res=(360, 180),
        loc=(0, 0),
    )

    features_setting = dict(
        feat_extractor=Shi_Tomasi_Extractor(maxCorners=200),
        tracker=LKTracker(),
        show_tracked_features=False,
    )

    kwargs = run_example(**scene_settings, **features_setting)
