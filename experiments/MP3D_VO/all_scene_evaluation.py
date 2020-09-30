from analysis.sequence_in_360_fov.all_solvers import *
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
            extra="ALL_POINTS",
            special_eval=True)
        initial_values = dict(
            iVal_Res_SK=(1, 1),
            iVal_Rpj_SK=(1, 1),
            iVal_Res_RtSK=(1, 1),
        )
        features_setting = dict(
            feat_extractor=Shi_Tomasi_Extractor(maxCorners=200),
            tracker=LKTracker(),
            show_tracked_features=False)

        log_settings = dict(log_files=(os.path.dirname(os.path.dirname(__file__)) +
                                       "/utilities/camera_recovering.py",),
                            filename=get_file_name(file_src=__file__,
                                                   **scene_settings, **features_setting,
                                                   **initial_values,
                                                   ),
                            save_bearings=False)
        kwargs = run_evaluation(**scene_settings, **features_setting,
                                **initial_values, **log_settings)

        plot_errors(**kwargs)
        plot_bar_errors(**kwargs)
        save_info(**kwargs)
