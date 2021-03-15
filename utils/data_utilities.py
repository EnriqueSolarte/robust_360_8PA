from config import Cfg
import os
from dataset_reader import MP3D_VO

def get_dataset(cfg: Cfg):
    if cfg.dataset == "MP3D_VO":
        dataset_dir = os.getenv("MP3D_VO_DATASET_DIR")
        scene = cfg.scene + "/" + cfg.scene_version
        return MP3D_VO(dt_dir=dataset_dir, scene=scene)
    elif cfg.dataset == "TUM_VI":
        print("Not yet tun VI!!!!")
        exit()


def get_image_mask(cfg: Cfg):
    """
    returns a mask map given a resolution res=(theta, phi) and location
    loc(theta, phi) camera orientation
    """
    shape = cfg.dataset
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
