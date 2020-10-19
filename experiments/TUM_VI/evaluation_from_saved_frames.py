from analysis.sequence_in_360_fov.all_solvers import *
from read_datasets.TUM_VI import TUM_VI
import os

if __name__ == '__main__':
    path = "/home/kike/Documents/datasets/TUM_VI/dataset"
    label = "_KS:L1_RT:L1_RTKS:a-L2_KS-RT:a-L2"
    extra = generate_fingerprint_time() + label
    for num in [1, 2, 3, 4, 5, 6]:
        scene = "dataset-room{}_512_16".format(num)
        data = TUM_VI(scene=scene, basedir=path)
        scene_settings = dict(
            data_scene=data,
            idx_frame=0,
            distance_threshold=0.5,
            extra=extra,
            special_eval=True,
            use_saved_bearings=True
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
            show_tracked_features=False,
            timing_evaluation=True,
            pinhole_model=True,
        )

        log_settings = dict(log_files=(os.path.dirname(os.path.dirname(__file__)) +
                                       "/utilities/camera_recovering.py",),
                            filename=get_file_name(file_src=__file__,
                                                   **scene_settings, **features_setting,
                                                   **initial_values,
                                                   ),
                            save_bearings=True)
        kwargs = run_evaluation(**scene_settings, **features_setting,
                                **initial_values, **log_settings)

        plot_bar_errors_and_time(**kwargs)
        save_info(**kwargs)
