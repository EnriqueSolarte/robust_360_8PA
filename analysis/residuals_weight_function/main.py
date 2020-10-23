from read_datasets.MP3D_VO import MP3D_VO
from structures.tracker import LKTracker
from structures.extractor.shi_tomasi_extractor import Shi_Tomasi_Extractor
import os
from analysis.utilities.data_utilities import get_bearings_by_plc
from analysis.utilities.data_utilities import get_bearings


def run_evaluation(**kwargs):
    while True:
        if kwargs.get("sampling_plc", True):
            kwargs, ret = get_bearings_by_plc(**kwargs)
        else:
            kwargs, ret = get_bearings(**kwargs)
        if not ret:
            break



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
        linear_motion=(-1, 1),
        angular_motion=(-25, 25),
        res=(360, 180),
        loc=(0, 0),
        skip_frames=5,
        noise=10000,
        inliers_ratio=1.0
    )
    initial_values = dict(
        iVal_Res_SK=(1, 1),
    )
    features_setting = dict(
        feat_extractor=Shi_Tomasi_Extractor(maxCorners=1000),
        tracker=LKTracker(),
        show_tracked_features=False,
        sampling=1000,
        timing_evaluation=True,
        extra="test"
        # extra=generate_fingerprint_time(),
    )

    log_settings = dict(
        log_files=(os.path.dirname(os.path.dirname(__file__)) +
                   "/utilities/camera_recovering.py",))

    kwargs = run_evaluation(**scene_settings, **features_setting,
                            **initial_values, **log_settings)
