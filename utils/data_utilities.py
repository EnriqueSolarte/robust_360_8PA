from config import Cfg
import os
from dataset_reader.MP3D_VO import MP3D_VO
from dataset_reader.TUM_VI import TUM_VI

def get_dataset(cfg: Cfg):
    if cfg.dataset_name == "MP3D_VO":
        dataset_dir = cfg.DIR_MP3D_VO_DATASET
        scene = cfg.scene + "/" + cfg.scene_version
        cfg.dataset =  MP3D_VO(dt_dir=dataset_dir, scene=scene)
        return cfg.dataset


    elif cfg.dataset_name == "TUM_VI":
        dataset_dir = cfg.DIR_TUM_VI_DATASET
        scene = "dataset-room{}_{}".format(cfg.scene, cfg.scene_version)
        cfg.dataset =  TUM_VI(dt_dir=dataset_dir, scene=scene)
        return cfg.dataset


def get_image_mask(cfg: Cfg):
    """
    returns a mask map given a resolution res=(theta, phi) and location
    loc(theta, phi) for camera orientation
    """
    NotImplementedError
    shape = cfg.dataset_name
    assert len(shape) == 2
    from geometry_utilities import extend_array_to_homogeneous as ext

    h, w = shape
    theta = (-res[0] / 2 + loc[0]), res[0] / 2 + loc[0]
    phi = -res[1] / 2 + loc[1], res[1] / 2 + loc[1]
    # ! (theta, phi) = Kinv * (u, v)
    K = np.linalg.inv(
        np.asarray((2 * np.pi / w, 0, -np.pi, 0, -np.pi / h, np.pi / 2, 0, 0,
                    1)).reshape(3, 3))
    sph_coord = np.radians(np.vstack((theta, phi)))
    uv_coord = K.dot(ext(sph_coord)).astype(int)
    mask = np.zeros((h, w))
    mask[uv_coord[1, 1]:uv_coord[1, 0], uv_coord[0, 0]:uv_coord[0, 1]] = 1
    return (mask * 255).astype(np.uint8)


def save_bearings(*args, **kwargs):

    # In case that the bearing vector were not calculated the whole 
    # fucntion is skiped
    if kwargs["bearings_kf"] is not None:
        dt = pd.DataFrame(np.vstack((kwargs["bearings"]["kf"], kwargs["bearings"]["frm"])).T)
        dirname = os.path.join(os.path.dirname(os.path.dirname(kwargs["filename"])),
                               "saved_bearings",
                               "saving_tracked_features" +
                               "_samples_" + str(kwargs["sampling"]) +
                               "_dist_" + str(kwargs["distance_threshold"]) +
                               "_res_" + str(kwargs["res"][0]) + "." + str(kwargs["res"][1]) +
                               "_loc_" + str(kwargs["loc"][0]) + "." + str(kwargs["loc"][1]),
                               "frames",
                               )
        file_bearings = str(kwargs["tracker"].initial_frame.idx) + "_" + str(
            kwargs["tracker"].tracked_frame.idx) + ".txt"
        file_bearings = os.path.join(dirname, file_bearings)
        print("scene:{}".format(kwargs["data_scene"].scene))
        print("Frames Kf:{}-frm:{}".format(kwargs["tracker"].initial_frame.idx, kwargs["tracker"].tracked_frame.idx))
        print("tracked features {}".format(kwargs["bearings"]["kf"].shape[1]))
        create_dir(dirname, delete_previous=False)
        print(file_bearings)
        dt.to_csv(file_bearings, header=None, index=None)