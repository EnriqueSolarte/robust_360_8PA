from read_datasets.MP3D_VO import MP3D_VO
from structures.tracker import LKTracker
from structures.extractor.shi_tomasi_extractor import Shi_Tomasi_Extractor
from file_utilities import generate_fingerprint_time
from analysis.utilities.plot_and_save_utilities import get_file_name, save_bearings
from analysis.utilities.data_utilities import save_bearings_vectors
import os

if __name__ == '__main__':
    path = "/home/kike/Documents/datasets/MP3D_VO"
    scene_list = os.listdir(path)
    for sc in scene_list:
        scene = sc + "/0"
        data = MP3D_VO(scene=scene, basedir=path)
        scene_settings = dict(
            data_scene=data,
            idx_frame=0,
            distance_threshold=0.5,
            res=(360, 180),
            loc=(0, 0),
            extra="used_features",
            special_eval=True)

        features_setting = dict(
            feat_extractor=Shi_Tomasi_Extractor(maxCorners=1000),
            tracker=LKTracker(),
            show_tracked_features=False,
        )
        log_settings = dict(filename=get_file_name(file_src=__file__,
                                                   **scene_settings, **features_setting,
                                                   ),
                            save_bearings=False)
        save_bearings_vectors(**scene_settings,
                              **features_setting,
                              **log_settings)
