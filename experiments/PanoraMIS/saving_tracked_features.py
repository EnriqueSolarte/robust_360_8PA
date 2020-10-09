from read_datasets.TUM_VI import TUM_VI
from structures.tracker import LKTracker
from structures.extractor.shi_tomasi_extractor import Shi_Tomasi_Extractor
from file_utilities import generate_fingerprint_time
from analysis.utilities.plot_and_save_utilities import get_file_name, save_bearings
import cv2
import os
from analysis.utilities.data_utilities import save_bearings_vectors

if __name__ == '__main__':

    if __name__ == '__main__':
        path = "/home/kike/Documents/datasets/TUM_VI/dataset"
        scene_list = os.listdir(path)

        # scene = "dataset-room3_512_16"
        scene = "dataset-room1_512_16"

        data = TUM_VI(scene=scene, basedir=path)
        scene_settings = dict(
            data_scene=data,
            idx_frame=531,
            distance_threshold=0.5,
            extra=generate_fingerprint_time(),
            special_eval=True,
            sampling=200
        )
        initial_values = dict(
            iVal_Res_SK=(1, 1),
            iVal_Res_RtSK=(1, 1),
        )
        features_setting = dict(
            feat_extractor=Shi_Tomasi_Extractor(
                maxCorners=1000,
                qualityLevel=0.001,
                minDistance=20,
                blockSize=1
            ),
            tracker=LKTracker(
                skip_frames=1,
                lk_params=dict(winSize=(15, 15),
                               maxLevel=2,
                               criteria=(cv2.TERM_CRITERIA_EPS
                                         | cv2.TERM_CRITERIA_COUNT, 20, 0.01))),
            show_tracked_features=True,
            timing_evaluation=True,
            mask_in_all_image=True,
        )

        log_settings = dict(filename=get_file_name(file_src=__file__,
                                                   **scene_settings, **features_setting,
                                                   ))
        save_bearings_vectors(**scene_settings,
                              **features_setting,
                              **log_settings)
