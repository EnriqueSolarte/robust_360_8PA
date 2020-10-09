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

        features_setting = dict(
            feat_extractor=Shi_Tomasi_Extractor(maxCorners=1000),
            tracker=LKTracker(),
            show_tracked_features=False,
            special_eval=True,
            distance_threshold=0.5
        )

        ransac_settings = dict(
            expected_inliers=1,
            probability_success=0.99,
            residual_threshold=threshold,
            post_function_evaluation=get_e_by_opt_res_error_SK_Rt,
            extra="KS_Rt",
            timing_evaluation=True,
        )

        kwargs = eval_ransac_8pa(**scene_settings, **ransac_settings, **features_setting)
        # plot_errors(**kwargs)
        # plot_time_results(**kwargs)
        plot_bar_errors(**kwargs)
        save_info(**kwargs)
