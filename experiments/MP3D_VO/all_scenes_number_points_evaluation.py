from analysis.sequence_in_360_fov.all_solvers import *

import os

if __name__ == '__main__':
    path = "/home/kike/Documents/datasets/MP3D_VO"
    scene_list = os.listdir(path)
    for sc in scene_list[4:]:
        # if sc in ("i5noydFURQK", "sT4fr6TAbpF"):
        #     continue
        scene = sc + "/0"
        data = MP3D_VO(scene=scene, basedir=path)
        scene_settings = dict(
            data_scene=data,
            idx_frame=0,
            distance_threshold=0.5,
            res=(360, 180),
            loc=(0, 0),
            extra="pts_eval",
            special_eval=True)

        features_setting = dict(
            feat_extractor=Shi_Tomasi_Extractor(maxCorners=1000),
            tracker=LKTracker(),
            show_tracked_features=False)

        for sampling in range(10, 201, 10):
            initial_values = dict(
                iVal_Res_SK=(1, 1),
                iVal_Res_RtSK=(1, 1),
                sampling=sampling,
                timing_evaluation=True
            )
            log_settings = dict(log_files=(
                "/home/kike/Documents/Research/optimal8PA/analysis/utilities/camera_recovering.py",),
                filename=get_file_name(file_src=__file__,
                                       **scene_settings,
                                       **features_setting,
                                       **initial_values,
                                       ),
                save_bearings=True)

            kwargs = run_evaluation(
                **scene_settings, **features_setting, **initial_values, **log_settings
            )
            plot_bar_errors_and_time(**kwargs)
            save_info(**kwargs)
