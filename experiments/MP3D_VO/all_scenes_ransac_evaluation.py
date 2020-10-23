from analysis.ransac_evaluations.opt_ransac_eval import *
from file_utilities import generate_fingerprint_time
from analysis.utilities.save_and_load_data import load_bearings_from_sampled_pcl, load_bearings_from_tracked_features


def run_ransac_evaluation(**kwargs):
    # ! In case of filename has nor been defined yet
    if "filename" not in kwargs.keys():
        kwargs["filename"] = get_file_name(**kwargs, file_src=__file__)
    while True:
        if kwargs.get("use_synthetic_points", True):
            kwargs, ret = load_bearings_from_sampled_pcl(**kwargs)
        else:
            kwargs, ret = load_bearings_from_tracked_features(**kwargs)

        if not ret:
            break
        print(
            "================================================================="
        )
        print("{}".format(kwargs["filename"]))
        # ! Based on RESIDUALS
        kwargs = default_ransac_8pa(**kwargs)
        kwargs = ransac_8pa_with_opt_rt_L2(**kwargs)
        kwargs = ransac_8pa_with_opt_ks(**kwargs)
        kwargs = ransac_8pa_with_opt_ksrt(**kwargs)

    return kwargs


if __name__ == '__main__':
    path = "/home/kike/Documents/datasets/MP3D_VO"
    scene_list = os.listdir(path)

    keyword = "samples_400_inliers_0.5_noise_500"
    # keyword = "samples_200_dist"
    extra = "_(RT:L2)-(KS:L1)-(RTKS:L2)_{}_{}".format(keyword, generate_fingerprint_time())

    for sc in scene_list:
        scene = sc + "/0"
        data = MP3D_VO(scene=scene, basedir=path)
        scene_settings = dict(
            data_scene=data,
            idx_frame=0,
            use_synthetic_points=True,
            extra=extra,
            keyword=keyword
        )

        ransac_settings = dict(
            expected_inliers=0.5,
            probability_success=0.99,
            residual_threshold_8PA=1.5e-3,
            relaxed_threshold_1=1e-4,
            residual_threshold_2=2e-5,
            min_super_set=50
        )

        opt_setting = dict(
            filename=get_file_name(file_src=__file__,
                                   **scene_settings,
                                   **ransac_settings,
                                   )
        )

        kwargs = run_ransac_evaluation(
            **scene_settings,
            **ransac_settings,
            **opt_setting,
        )

        plot_bar_errors_and_time(**kwargs)
        save_info(**kwargs)
