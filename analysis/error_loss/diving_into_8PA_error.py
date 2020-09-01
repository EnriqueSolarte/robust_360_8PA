from solvers.epipolar_constraint import EightPointAlgorithmGeneralGeometry
from read_datasets.MP3D_VO import get_default_mp3d_scene
from structures.tracker import LKTracker
from structures.extractor.shi_tomasi_extractor import Shi_Tomasi_Extractor
from analysis.utilities import track_features
from image_utilities import get_mask_map_by_res_loc
from solvers.epipolar_constraint_by_ransac import RansacEssentialMatrix


def eval_function(**kwargs):
    kwargs["mask"] = get_mask_map_by_res_loc(kwargs["data_scene"].shape,
                                             res=kwargs["res"],
                                             loc=kwargs["loc"])
    bearings_kf, bearings_frm, cam_gt, kwargs = track_features(**kwargs)

    cam_8pa = RansacEssentialMatrix(**kwargs).solve(data=(bearings_kf.T, bearings_frm.T), solver="g8p")
    print('done')


if __name__ == '__main__':
    data = get_default_mp3d_scene()

    scene_settings = dict(
        data_scene=data,
        idx_frame=549,
        distance_threshold=0.5,
        res=(360, 180),
        loc=(0, 0),
    )

    ransac_parm = dict(min_samples=8,
                       p_succes=0.99,
                       outliers=0.5,  # * expecting 50% of outliers
                       residual_threshold=0.1,
                       verbose=True)

    features_setting = dict(
        feat_extractor=Shi_Tomasi_Extractor(),
        tracker=LKTracker(),
        show_tracked_features=True
    )

    model_settings = dict(
        opt_version="v1",
    )
    eval_function(**scene_settings,
                  **model_settings,
                  **features_setting,
                  **ransac_parm)
