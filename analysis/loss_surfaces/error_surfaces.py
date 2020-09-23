from read_datasets.MP3D_VO import MP3D_VO
from structures.tracker import LKTracker
from structures.extractor.shi_tomasi_extractor import Shi_Tomasi_Extractor
from analysis.utilities.surfaces_utilities import *


def eval_frame(**kwargs):
    kwargs["filename"] = get_file_name(**kwargs, file_src=__file__)

    kwargs, _ = get_bearings(**kwargs)
    kwargs = create_grid(**kwargs)
    kwargs = get_eval_of_8PA(**kwargs)
    bearings_kf = kwargs["bearings"]["kf"].copy()
    bearings_frm = kwargs["bearings"]["frm"].copy()
    g8p = EightPointAlgorithmGeneralGeometry()
    for idx in kwargs["grid_range"]:
        s = kwargs["ss_grid"][idx]
        k = kwargs["kk_grid"][idx]
        kwargs["bearings"]["kf_norm"], t1 = normalizer_s_k(
            x=bearings_kf, s=s, k=k)
        kwargs["bearings"]["frm_norm"], t2 = normalizer_s_k(
            x=bearings_frm, s=s, k=k)

        kwargs['e_norm'] = g8p.compute_essential_matrix(
            x1=kwargs["bearings"]["kf_norm"],
            x2=kwargs["bearings"]["frm_norm"],
        )
        kwargs['e_hat'] = t1.T @ kwargs['e_norm'].copy() @ t2
        kwargs['cam_hat'] = g8p.recover_pose_from_e(
            E=kwargs['e_hat'], x1=bearings_kf, x2=bearings_frm)
        kwargs = eval_surfaces(**kwargs)
        print("{} : {}".format(kwargs["filename"],
                               idx / kwargs["ss_grid"].size))
    return kwargs


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
        # idx_frame=85,
        distance_threshold=0.5,
        res=(360, 180),
        # res=(180, 180),
        # res=(65.5, 46.4),
        loc=(0, 0),
        grid=(-1, 1, 50),
        extra="samplings",
    )

    features_setting = dict(
        feat_extractor=Shi_Tomasi_Extractor(maxCorners=1000),
        sampling=8,
        tracker=LKTracker(),
        show_tracked_features=False)

    ransac_parm = dict(
        min_samples=8,
        max_trials=RansacEssentialMatrix.get_number_of_iteration(
            p_success=0.99, outliers=0.5, min_constraint=8),
        residual_threshold=1e-5,
        verbose=True,
        use_ransac=False)

    kwargs = eval_frame(**scene_settings, **features_setting, **ransac_parm)

    plot_surfaces(**kwargs)
    save_surfaces(**kwargs)
