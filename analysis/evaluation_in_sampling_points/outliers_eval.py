from analysis.evaluation_in_sampling_points.all_solvers import *
import os

if __name__ == '__main__':
    path = "/home/kike/Documents/datasets/MP3D_VO"
    scene = "2azQ1b91cZZ/0"
    for inliers_ratio in (1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4):
        data = MP3D_VO(scene=scene, basedir=path)
        scene_settings = dict(
            data_scene=data,
            idx_frame=0,
            linear_motion=(-1, 1),
            angular_motion=(-10, 10),
            res=(360, 180),
            loc=(0, 0),
            extra="inliers_eval",
            skip_frames=1,
            noise=500,
            inliers_ratio=inliers_ratio,
            sampling=200,
        )
        initial_values = dict(
            iVal_Res_SK=(1, 1),
            iVal_Rpj_SK=(1, 1),
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

        plot_errors(**kwargs)
        plot_bar_errors(**kwargs)
        save_info(**kwargs)
