from read_datasets.MP3D_VO import MP3D_VO
from structures.tracker import LKTracker
from structures.extractor.shi_tomasi_extractor import Shi_Tomasi_Extractor
from analysis.utilities.surfaces_utilities import *
from file_utilities import generate_fingerprint_time


def eval_frame(**kwargs):
    kwargs["random_state"] = np.random.RandomState(1000)

    kwargs["filename"] = get_file_name(**kwargs, file_src=__file__)
    if not kwargs.get("use_sampling_points", False):
        kwargs, _ = get_bearings(**kwargs)
    else:
        kwargs, _ = get_bearings_by_plc(**kwargs)

    kwargs["bearings"]["frm"] = add_outliers_to_pcl(
        kwargs["bearings"]["frm"].copy(),
        inliers=int(kwargs.get("inliers_ratio", 0.5) * kwargs["bearings"]["frm"].shape[1]))

    kwargs["bearings"]["frm"] = sph.sphere_normalization(kwargs["bearings"]["frm"])
    kwargs = create_grid(**kwargs)
    kwargs = get_eval_of_8PA(**kwargs)
    bearings_kf = kwargs["bearings"]["kf"].copy()
    bearings_frm = kwargs["bearings"]["frm"].copy()
    g8p = EightPointAlgorithmGeneralGeometry()
    for idx in kwargs["grid_range"]:
        s = kwargs["ss_grid"][idx]
        k = kwargs["kk_grid"][idx]
        kwargs["bearings"]["kf_norm"], t1 = normalizer_s_k(x=bearings_kf,
                                                           s=s,
                                                           k=k)
        kwargs["bearings"]["frm_norm"], t2 = normalizer_s_k(x=bearings_frm,
                                                            s=s,
                                                            k=k)

        kwargs['e_norm'] = g8p.compute_essential_matrix(
            x1=kwargs["bearings"]["kf_norm"],
            x2=kwargs["bearings"]["frm_norm"],
        )
        kwargs['e_hat'] = t1.T @ kwargs['e_norm'].copy() @ t2
        kwargs['cam_hat'] = g8p.recover_pose_from_e(E=kwargs['e_hat'],
                                                    x1=bearings_kf,
                                                    x2=bearings_frm)
        kwargs = eval_surfaces(**kwargs)
        print("{} : {}".format(idx / kwargs["ss_grid"].size, kwargs["filename"]))
    return kwargs


if __name__ == '__main__':
    path = "/home/kike/Documents/datasets/MP3D_VO"
    scene = "i5noydFURQK/0"
    # scene = "1LXtFkjw3qL/0"
    # scene = "759xd9YjKW5/0"
    # path = "/run/user/1001/gvfs/sftp:host=140.114.27.95,port=50002/NFS/kike/minos/vslab_MP3D_VO/512x1024"
    data = MP3D_VO(scene=scene, basedir=path)
    label_info = "icra2021-{}".format(generate_fingerprint_time())
    np.random.seed(100)
    scene_settings = dict(
        data_scene=data,
        idx_frame=370,
        linear_motion=(0.1, 1),
        angular_motion=(-45, 45),
        res=(180, 180),
        loc=(0, 0),
        extra=label_info,
        skip_frames=1,
        noise=10,
        inliers_ratio=1,
        sampling=100,
        distance_threshold=0.5,
        grid=(-1, 1, 50),
        use_sampling_points=True
    )

    features_setting = dict(
        feat_extractor=Shi_Tomasi_Extractor(maxCorners=250,
                                            qualityLevel=0.1,
                                            minDistance=20,
                                            blockSize=5),
        tracker=LKTracker(lk_params=dict(winSize=(15, 15),
                                         maxLevel=4,
                                         criteria=(cv2.TERM_CRITERIA_EPS
                                                   | cv2.TERM_CRITERIA_COUNT, 10, 0.1))),
        show_tracked_features=True)

    kwargs = eval_frame(**scene_settings, **features_setting)

    plot_surfaces(**kwargs)
    save_surfaces(**kwargs)
