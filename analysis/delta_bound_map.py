from delta_bound import eval_delta_bound_by_fov
import numpy as np
import matplotlib.pyplot as plt


def delta_map(norm=False):
    v = np.linspace(1, 180, 180)
    h = np.linspace(1, 360, 360)
    hh, vv = np.meshgrid(h, v)
    d_map_ = np.zeros_like(hh).flatten()
    ind = 0
    for v_fov, h_fov in zip(vv.flatten(), hh.flatten()):
        cfg = dict(v_fov=v_fov,
                   h_fov=h_fov,
                   norm=norm)
        d_map_[ind] = eval_delta_bound_by_fov(**cfg)

        print("resolution: {0:}x{1:}  - d:{2:1.2f}  - progress{3:0.2f}".format(v_fov,
                                                                               h_fov,
                                                                               d_map_[ind],
                                                                               ind / d_map_.size))
        ind += 1
    return vv, hh, d_map_.reshape(hh.shape)


if __name__ == '__main__':
    h, v, d_map = delta_map(norm=True)
    plt.contourf(v, h, d_map, 20, cmap='viridis')
    plt.colorbar()
    plt.title("Delta-bound map for 8PA norm")
    plt.xlabel("horizontal FOV")
    plt.ylabel("vertical FOV")
    plt.show()
