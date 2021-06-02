from imageio.core.util import has_module
from config import Cfg
from utils import *
from solvers import *
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


def plot_sampling_evaluations(cfg: Cfg):
    sampler = BearingsSampler(cfg)

    list_methods = [
        get_cam_pose_by_8pa,
        get_cam_pose_by_opt_SK,
        get_cam_pose_by_GSM,
        get_cam_pose_by_GSM_const_wRT,
        get_cam_pose_by_GSM_const_wSK
    ]

    hist_errors = {}

    while True:
        data_bearings, ret = sampler.get_bearings(return_dict=True)

        if not ret:
            break
        bearings_kf = data_bearings["bearings_kf"]
        bearings_frm = data_bearings["bearings_frm"]
        cam_pose_gt = data_bearings["relative_pose"]
        if bearings_kf is not None:
            if bearings_kf.shape[1] < 8:
                continue

            print("#\n")
            print("Number of bearings evaluated: {}".format(bearings_kf.shape[1]))
            for method in list_methods:
                cam_pose_hat = method(
                    x1=bearings_kf,
                    x2=bearings_frm
                )

                cam_pose_error = evaluate_error_in_transformation(
                    transform_est=cam_pose_hat,
                    transform_gt=cam_pose_gt
                )
                hist_errors = add_error_eval(cam_pose_error, method.__name__, hist_errors)

        # save_bearings(**data_bearings, save_config=True,
        #               save_camera_as=cfg.CAM_POSES_GT)

        # save_results(errors=hist_errors, cfg=cfg)
        plot_evaluations(hist_errors, cfg)


def plot_evaluations(hist_eval, cfg: Cfg):
    number_of_iterations = hist_eval[Cfg._8PA][0].__len__()
    iterations = np.linspace(0, number_of_iterations-1, number_of_iterations)
    plt.figure("Sampling Evaluations", figsize=(20, 5))
    plt.clf()

    ax1 = plt.subplot(231)
    plt.title("8-PA vs. $Opt~\epsilon(S,K)$")
    plt.plot(iterations, hist_eval[Cfg._8PA + "_rot_e"],
             label="8-PA",
             c=mcolors.TABLEAU_COLORS["tab:blue"],
             linestyle=":",
             marker="."
             )
    plt.plot(iterations, hist_eval[Cfg.OPT_eSK + "_rot_e"],
             label="$Opt~\epsilon(S,K)$ (Ours)",
             c="orange",
             marker="."
             )
    plt.ylabel("Rot-e (MAE)")
    plt.legend()
    plt.grid()
    plt.setp(ax1.get_xticklabels(), visible=False)
    ax2 = plt.subplot(234, sharex=ax1)
    plt.plot(iterations, hist_eval[Cfg._8PA + "_tra_e"],
             label="8-PA",
             c=mcolors.TABLEAU_COLORS["tab:blue"],
             linestyle=":",
             marker="."
             )
    plt.plot(iterations, hist_eval[Cfg.OPT_eSK + "_tra_e"],
             label="$Opt~\epsilon(S,K)$ (Ours)",
             c="orange",
             marker="."
             )
    plt.xlabel("Number of Evaluated Frames")
    plt.ylabel("Trans-e (MAE)")
    plt.legend()
    plt.grid()

    ax3 = plt.subplot(232, sharey=ax1)
    plt.title("GSM vs. $\omega_{SK}$")
    plt.plot(iterations, hist_eval[Cfg.GSM + "_rot_e"],
             label="GSM",
             c=mcolors.TABLEAU_COLORS["tab:brown"],
             linestyle=":",
             marker="."
             )
    # plt.plot(iterations, hist_eval[Cfg.wRT + "_rot_e"], label="$\omega_{RT}$", c=mcolors.TABLEAU_COLORS["tab:red"])
    plt.plot(iterations, hist_eval[Cfg.wSK + "_rot_e"],
             label="$\omega_{SK}$ (Ours)",
             c=mcolors.TABLEAU_COLORS["tab:green"],
             marker="."
             )
    plt.ylabel("Rot-e (MAE)")
    plt.legend()
    plt.grid()
    plt.setp(ax3.get_xticklabels(), visible=False)
    plt.subplot(235, sharey=ax2)
    plt.plot(iterations, hist_eval[Cfg.GSM + "_tra_e"],
             label="GSM",
             c=mcolors.TABLEAU_COLORS["tab:brown"],
             linestyle=":",
             marker="."
             )
    # plt.plot(iterations, hist_eval[Cfg.wRT + "_tra_e"], label="$\omega_{RT}$", c=mcolors.TABLEAU_COLORS["tab:red"])
    plt.plot(iterations, hist_eval[Cfg.wSK + "_tra_e"],
             label="$\omega_{SK}$ (Ours)",
             c=mcolors.TABLEAU_COLORS["tab:green"],
             marker="."
             )

    plt.xlabel("Number of Evaluated Frames")
    plt.ylabel("Trans-e (MAE)")
    plt.legend()
    plt.grid()

    msg = "Number of correspondence:{}\nNoise vMF $\kappa$:{}\nOutliers: ${}$%\nEvaluation:{}".format(
        cfg.params.max_number_features,
        cfg.params.vMF_kappa, cfg.params.outliers_ratio*100,
        number_of_iterations
    )
    plt.gcf().text(0.64, 0.12, msg, size=10)
    plt.draw()
    plt.waitforbuttonpress(0.001)
    plt.show(block=False)


if __name__ == '__main__':

    config_file = Cfg.FILE_CONFIG_MP3D_VO
    cfg = Cfg.from_cfg_file(yaml_config=config_file)
    plot_sampling_evaluations(cfg)
