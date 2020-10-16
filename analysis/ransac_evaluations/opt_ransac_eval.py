from analysis.ransac_evaluations.default_ransac_eval import *


def OPT_ransac_opt_rt(**kwargs):
    ransac = RANSAC_OPT_8PA(**kwargs)
    ransac.prior_function_evaluation = get_e_by_opt_res_error_Rt
    ransac.post_function_evaluation = get_e_by_opt_res_error_Rt
    ransac.min_super_set = kwargs.get("min_super_set", 30)
    return eval(method="8PA_Rt", ransac=ransac, **kwargs)


def OPT_ransac_opt_ks(**kwargs):
    ransac = RANSAC_OPT_8PA(**kwargs)
    ransac.prior_function_evaluation = get_e_by_opt_res_error_Rt
    ransac.post_function_evaluation = get_e_by_opt_res_error_SK
    ransac.min_super_set = kwargs.get("min_super_set", 30)
    return eval(method="opt_ks", ransac=ransac, **kwargs)


def OPT_ransac_opt_ks_rt(**kwargs):
    ransac = RANSAC_OPT_8PA(**kwargs)

    ransac.prior_function_evaluation = get_e_by_opt_res_error_Rt
    ransac.post_function_evaluation = get_e_by_opt_res_error_SK_Rt
    ransac.min_super_set = kwargs.get("min_super_set", 30)
    return eval(method="opt_ks_rt", ransac=ransac, **kwargs)


def eval_methods_opt_ransac(**kwargs):
    if "filename" not in kwargs.keys():
        kwargs["filename"] = get_file_name(**kwargs, file_src=__file__)

    while True:
        kwargs, ret = get_bearings(**kwargs)
        if not ret:
            break
        print("==================================================")
        kwargs = default_ransac_8pa(**kwargs)
        kwargs = OPT_ransac_opt_rt(**kwargs)
        kwargs = OPT_ransac_opt_ks(**kwargs)
        kwargs = OPT_ransac_opt_ks_rt(**kwargs)
        kwargs = eval_cam_pose_error(**kwargs)

        if kwargs.get("save_results", False):
            save_data(**kwargs)

        print(kwargs["filename"])
        print(kwargs["extra"])


if __name__ == '__main__':
    path = "/home/kike/Documents/datasets/MP3D_VO"
    # scene = "2azQ1b91cZZ/0"
    # scene = "i5noydFURQK/0"
    scene = "sT4fr6TAbpF/0"
    # scene = "1LXtFkjw3qL/0"
    # scene = "759xd9YjKW5/0"

    data = MP3D_VO(scene=scene, basedir=path)
    scene_settings = dict(
        data_scene=data,
        idx_frame=0,
        res=(360, 180),
        loc=(0, 0),
        sampling=200,
        extra=generate_fingerprint_time()
    )
    features_setting = dict(
        feat_extractor=Shi_Tomasi_Extractor(maxCorners=1000),
        tracker=LKTracker(),
        show_tracked_features=False,
        special_eval=True)

    ransac_settings = dict(
        expected_inliers=0.5,
        probability_success=0.99,
        residual_threshold_8PA=3e-5,
        residual_threshold=2e-5,
        relaxed_threshold=2e-4,
        min_super_set=50
    )

    opt_setting = dict(
        save_results=True,
    )

    eval_methods_opt_ransac(
        **scene_settings,
        **ransac_settings,
        **features_setting,
        **opt_setting,
    )
