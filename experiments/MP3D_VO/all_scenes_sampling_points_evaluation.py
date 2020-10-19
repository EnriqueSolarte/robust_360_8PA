from analysis.evaluation_in_sampling_points.all_solvers import *
import os

if __name__ == '__main__':
    path = "/home/kike/Documents/datasets/MP3D_VO"
    label = "_RT:L2_KS:-L1_KS-RT:a.B=0.5-L1-RTKS:a.B=0.5-L2"
    # label = "_RT:L1_RTKS:B=0.3-a-L2_"
    extra = "_inliers-eval_" + generate_fingerprint_time() + label
    scene_list = os.listdir(path)
    for sc in scene_list:
        for inliers in (0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0):
        # for noise in (10, 100, 200, 500, 1000, 10000):
            scene = sc + "/0"
            data = MP3D_VO(scene=scene, basedir=path)
            scene_settings = dict(
                data_scene=data,
                idx_frame=0,
                linear_motion=(-1, 1),
                angular_motion=(-10, 10),
                res=(360, 180),
                loc=(0, 0),
                extra=extra,
                skip_frames=1,
                noise=500,
                inliers_ratio=inliers,
                sampling=200,
                save_bearings=True
            )
            initial_values = dict(
                iVal_Res_SK=(1, 1),
                iVal_Res_RtSK=(1, 1),
                timing_evaluation=True,
            )

            log_settings = dict(log_files=(os.path.dirname(os.path.dirname(__file__)) +
                                           "/utilities/camera_recovering.py",),
                                filename=get_file_name(file_src=__file__,
                                                       **scene_settings,
                                                       **initial_values,
                                                       ),
                                )

            kwargs = run_evaluation(**scene_settings, **initial_values, **log_settings)

            # plot_errors(**kwargs)
            plot_bar_errors_and_time(**kwargs)
            save_info(**kwargs)
