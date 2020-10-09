from analysis.ransac_evaluations.ransac_8pa_evaluation import *

if __name__ == '__main__':
    path = "/home/kike/Documents/datasets/MP3D_VO"
    scene = "2azQ1b91cZZ/0"
    # scene = "i5noydFURQK/0"
    # scene = "sT4fr6TAbpF/0"
    # scene = "1LXtFkjw3qL/0"
    # scene = "759xd9YjKW5/0"
    list_threshold = (1e-5, 1e-3)
    for threshold in list_threshold:
        data = MP3D_VO(scene=scene, basedir=path)
        scene_settings = dict(
            data_scene=data,
            idx_frame=0,
            res=(360, 180),
            loc=(0, 0),
            sampling=200,
        )

        initial_values = dict(
            iVal_Res_SK=(1, 1),
            iVal_Rpj_SK=(1, 1),
            iVal_Res_RtSK=(1, 1),
        )

        features_setting = dict(
            feat_extractor=Shi_Tomasi_Extractor(maxCorners=500),
            tracker=LKTracker(),
            show_tracked_features=False,
            special_eval=False,
            distance_threshold=0.5
        )

        ransac_settings = dict(
            expected_inliers=0.5,
            probability_success=0.99,
            residual_threshold=threshold,
            post_function_evaluation=get_e_by_8pa,
            # post_function_evaluation=get_e_by_opt_res_error_Rt,
            # post_function_evaluation=get_e_by_opt_res_error_SK_Rt,
            timing_evaluation=True,
        )

        log_settings = dict(
            filename=get_file_name(file_src=__file__,
                                   **scene_settings,
                                   **initial_values,
                                   **features_setting,
                                   **ransac_settings
                                   )
        )
        kwargs = eval_ransac_8pa(
            **scene_settings,
            **initial_values,
            **features_setting,
            **ransac_settings,
            **log_settings
        )
        # plot_errors(**kwargs)
        # plot_time_results(**kwargs)
        plot_bar_errors(**kwargs)
        save_info(**kwargs)
