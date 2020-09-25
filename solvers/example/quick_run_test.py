from structures.extractor.shi_tomasi_extractor import Shi_Tomasi_Extractor
from structures.tracker import LKTracker
from read_datasets.MP3D_VO import MP3D_VO
from solvers.epipolar_constraint_by_ransac import RansacEssentialMatrix

if __name__ == '__main__':
    path = "/home/kike/Documents/datasets/MP3D_VO"
    scene = "2azQ1b91cZZ/0"
    # scene = "1LXtFkjw3qL/0"
    # scene = "759xd9YjKW5/0"
    # path = "/run/user/1001/gvfs/sftp:host=140.114.27.95,port=50002/NFS/kike/minos/vslab_MP3D_VO/512x1024"
    data = MP3D_VO(scene=scene, basedir=path)

    scene_settings = dict(
        data_scene=data,
        idx_frame=0,
        distance_threshold=0.5,
        res=(360, 180),
        loc=(0, 0),
    )

    ransac_parm = dict(
        min_samples=8,
        max_trials=RansacEssentialMatrix.get_number_of_iteration(
            p_success=0.99, outliers=0.5, min_constraint=8),
        residual_threshold=1e-5,
        verbose=True,
        use_ransac=True,
        extra="projected_distance",
    )

    features_setting = dict(feat_extractor=Shi_Tomasi_Extractor(),
                            tracker=LKTracker(),
                            show_tracked_features=True)

    eval_run(**scene_settings, **features_setting, **ransac_parm)
