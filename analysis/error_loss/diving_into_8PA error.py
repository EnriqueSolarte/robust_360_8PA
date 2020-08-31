from solvers.epipolar_constraint import EightPointAlgorithmGeneralGeometry
from read_datasets.MP3D_VO import get_default_mp3d_scene
from structures.tracker import LKTracker
from structures.extractor.shi_tomasi_extractor import Shi_Tomasi_Extractor


def eval_function(**kwargs):
    print("done")


if __name__ == '__main__':
    data = get_default_mp3d_scene()

    scene_settings = dict(
        data_scene=data,
        idx_frame=549,
        distance_threshold=0.5,
        res=(360, 180),
        loc=(0, 0),
    )

    features_setting = dict(
        feat_extractor=Shi_Tomasi_Extractor(),
        tracker=LKTracker(),
    )

    model_settings = dict(
        opt_version="v1",
    )
    eval_function(**scene_settings,
                  **model_settings,
                  **features_setting)
