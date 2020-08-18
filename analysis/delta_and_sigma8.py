from read_datasets.MP3D_VO import MP3D_VO
from pcl_utilities import *
from analysis.delta_bound import get_delta_bound_by_bearings
from solvers.optimal8pa import Optimal8PA as norm_8pa


def main(**arg):
    g8p_norm = norm_8pa()
    path = "/home/kike/Documents/datasets/MP3D_VO"
    # path = "/run/user/1001/gvfs/sftp:host=140.114.27.95,port=50002/NFS/kike/minos/vslab_MP3D_VO/512x1024"
    data = MP3D_VO(scene=arg["scene"], path=path)

    pcl_dense, pcl_dense_color, _ = data.get_dense_pcl(idx=arg["frame"])
    pcl_dense, mask = mask_pcl_by_res_and_loc(pcl=pcl_dense, res=arg["res"], loc=arg["loc"])
    samples = np.random.randint(0, pcl_dense.shape[1], arg["samples"])

    bearings_a, bearings_b, cam_a2b = get_bearings_from_pcl(pcl=pcl_dense[:, samples],
                                                            t_vector=arg["t_vector"],
                                                            rot_vector=arg["r_vector"],
                                                            noise=arg["noise"],
                                                            outliers=arg["outliers"] * len(samples))

    delta_, C = get_delta_bound_by_bearings(bearings_a, bearings_b)
    u, sigma, v = np.linalg.svd(g8p_norm.building_matrix_A(bearings_a, bearings_b))

    print("Sigma_8:{}".format(sigma[-2]))
    print("Delta:{}".format(delta_))


if __name__ == '__main__':
    parameters = dict(scene="1LXtFkjw3qL/1",
                      frame=100,
                      res=(55, 65),
                      loc=(0, 0),
                      samples=200,
                      noise=500,
                      outliers=0.05,
                      t_vector=(np.random.uniform(-0.5, 0.5),
                                np.random.uniform(-0.5, 0.5),
                                np.random.uniform(-0.5, 0.5)),
                      r_vector=(np.random.uniform(-10, 10),
                                np.random.uniform(-10, 10),
                                np.random.uniform(-10, 10)))

    main(**parameters)
