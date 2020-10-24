from read_datasets.MP3D_VO import MP3D_VO
from structures.tracker import LKTracker
from structures.extractor.shi_tomasi_extractor import Shi_Tomasi_Extractor
from file_utilities import generate_fingerprint_time
from analysis.utilities.plot_utilities import get_file_name
from analysis.utilities.save_and_load_data import save_bearings_vectors
import os
import cv2

if __name__ == '__main__':
    path = "/home/kike/Documents/datasets/MP3D_VO"
    scene_list = os.listdir(path)
    label_info = generate_fingerprint_time() + "_saved_tracked_features_"
    for sc in ("i5noydFURQK",):
        scene = sc + "/1"
        data = MP3D_VO(scene=scene, basedir=path)
        scene_settings = dict(
            data_scene=data,
            idx_frame=0,
            distance_threshold=0.5,
            res=(360, 180),
            loc=(0, 0),
            extra="used_features",
            special_eval=True,
            sampling=200
        )

        features_setting = dict(
            feat_extractor=Shi_Tomasi_Extractor(maxCorners=200,
                                                qualityLevel=0.00001,
                                                minDistance=7,
                                                blockSize=1),
            tracker=LKTracker(lk_params=dict(winSize=(20, 20),
                                             maxLevel=2,
                                             criteria=(cv2.TERM_CRITERIA_EPS
                                                       | cv2.TERM_CRITERIA_COUNT, 1, 0.0001))),
            show_tracked_features=True,
        )
        log_settings = dict(filename=get_file_name(file_src=__file__,
                                                   **scene_settings, **features_setting,
                                                   ),
                            save_bearings=False)
        save_bearings_vectors(**scene_settings,
                              **features_setting,
                              **log_settings)
