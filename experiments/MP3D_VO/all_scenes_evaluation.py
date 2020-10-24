from analysis.sequence_in_360_fov.all_solvers import *
import os

if __name__ == '__main__':
    path = "/home/kike/Documents/datasets/MP3D_VO"
    scene_list = os.listdir(path)
    # label = "_KS:L1_RT:L1_KS-RT:b-*L1"
    # label = "_RT:L2_KS:-L1_KS-RT:a.B=0.5-L1-RTKS:a.B=0.5-L2"
    # label = "_RT:L1_RTKS:B=0.3-a-L2_"
    label = "_test_for_you_guys_"
    # extra = generate_fingerprint_time() + label
    extra = label

    for sc in ("i5noydFURQK",):
        scene = sc + "/1"
        data = MP3D_VO(scene=scene, basedir=path)
        scene_settings = dict(
            data_scene=data,
            idx_frame=0,
            distance_threshold=0.5,
            res=(360, 180),
            loc=(0, 0),
            extra=extra,
            keyword="samples_200_dist",
            special_eval=True,
            use_saved_bearings=True
        )

        initial_values = dict(
            iVal_Res_SK=(1, 1),
            iVal_Res_RtSK=(1, 1),
        )
        features_setting = dict(
            feat_extractor=Shi_Tomasi_Extractor(maxCorners=1000),
            tracker=LKTracker(),
            show_tracked_features=False,
            sampling=200,
            timing_evaluation=True,
        )

        log_settings = dict(log_files=(os.path.dirname(os.path.dirname(__file__)) +
                                       "/utilities/camera_recovering.py",),
                            filename=get_file_name(file_src=__file__,
                                                   **scene_settings, **features_setting,
                                                   **initial_values,
                                                   ),
                            save_bearings=False)
        kwargs = run_evaluation(**scene_settings, **features_setting,
                                **initial_values, **log_settings)

        # plot_errors(**kwargs)
        plot_bar_errors_and_time(**kwargs)
        save_info(**kwargs)
