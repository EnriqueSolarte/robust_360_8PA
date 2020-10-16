from analysis.ransac_evaluations.opt_ransac_eval import *

if __name__ == '__main__':
    path = "/home/kike/Documents/datasets/MP3D_VO"
    scene_list = os.listdir(path)
    fingerprint = generate_fingerprint_time()
    for sc in ["pRbA3pwrgk9"]:
        scene = sc + "/0"
        data = MP3D_VO(scene=scene, basedir=path)
        scene_settings = dict(
            data_scene=data,
            idx_frame=0,
            res=(360, 180),
            loc=(0, 0),
            sampling=500,
            extra=fingerprint
        )
        features_setting = dict(
            feat_extractor=Shi_Tomasi_Extractor(maxCorners=1000),
            tracker=LKTracker(),
            show_tracked_features=False,
            special_eval=True)

        ransac_settings = dict(
            expected_inliers=0.5,
            probability_success=0.99,
            residual_threshold_8PA=1e-4,
            residual_threshold=2e-5,
            relaxed_threshold=1e-4,
            min_super_set=50
        )

        opt_setting = dict(
            save_results=True,
            filename=get_file_name(file_src=__file__,
                                   **scene_settings,
                                   **ransac_settings,
                                   **features_setting
                                   )
        )

        save_this_file(
            dir=os.path.dirname(opt_setting['filename']),
            filename=__file__
        )
        eval_methods_opt_ransac(
            **scene_settings,
            **ransac_settings,
            **features_setting,
            **opt_setting,
        )