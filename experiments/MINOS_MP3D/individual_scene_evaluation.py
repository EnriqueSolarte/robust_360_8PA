from analysis.sequence_of_frames.all_solvers import *
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
            extra="Initial eval",
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

        ransac_parm = dict(
            min_samples=8,
            max_trials=RansacEssentialMatrix.get_number_of_iteration(
                p_success=0.99, outliers=0.5, min_constraint=8),
            residual_threshold=1e-5,
            verbose=True,
            use_ransac=False)

        log_settings = dict(
            log_files=(os.path.dirname(os.path.dirname(__file__)) +
                       "/utilities/camera_recovering.py", ))

        kwargs = run_sequence(**scene_settings, **features_setting,
                              **ransac_parm, **initial_values, **log_settings)

        plot_errors(**kwargs)
        plot_bar_errors(**kwargs)
        save_info(**kwargs)
