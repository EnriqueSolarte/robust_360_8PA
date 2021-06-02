from config import Cfg
from utils import *
from solvers import *
import matplotlib.pyplot as plt


def main(cfg: Cfg):

    tracker = FeatureTracker(cfg)

    s = []
    plt.figure("Stability", figsize=(6, 4))
    plt.grid()    
    while True:
        bearings_kf, bearings_frm, cam_pose_gt, ret = tracker.track(verbose=False)
        if not ret:
            break

        if bearings_kf is not None:
            if bearings_kf.shape[1] < 8:
                continue

        A = G8PA.building_matrix_A(x1=bearings_kf, x2=bearings_frm)
        _, sigma, _ = np.linalg.svd(A)
        
        s.append(sigma[-2])

        plt.title("Stability")
        plt.plot(range(s.__len__()), s, c="red",marker=".")
        plt.xlabel("frame idx")
        plt.ylabel("stability $\sigma_8$")
        plt.draw()
        plt.waitforbuttonpress(0.01)


if __name__ == '__main__':
    config_file = Cfg.FILE_CONFIG_MP3D_VO
    # config_file = Cfg.FILE_CONFIG_TUM_VI
    cfg = Cfg.from_cfg_file(yaml_config=config_file)
    main(cfg)
