import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from config import *


def plot_histograms_of_errors():
    if experiment_group == "noise":
        res = ress[1]
        for noise in reversed(noises):
            dt = pd.read_csv(
                "../report/{}/{}/{}/{}/{}/{}/{}_{}_{}_{}_{}_{}_{}_{}.csv".
                format(scene, str(idx_frame),
                       "mc" if motion_constraint else "!mc", noise,
                       str(res[0]) + 'x' + str(res[1]), pts, scene[:-2],
                       scene[-1:], str(idx_frame),
                       "mc" if motion_constraint else "!mc", noise,
                       str(res[0]) + 'x' + str(res[1]), pts, opt_version))

            headers = dt.columns.values
            data = dt.values

            plt.figure()
            plt.subplot(1, 2, 1)
            plt.hist(data[:, 0],
                     density=True,
                     bins=data.shape[0] // 10,
                     label=headers[0])
            plt.hist(data[:, 2],
                     density=True,
                     bins=data.shape[0] // 10,
                     label=headers[2])
            plt.legend()
            plt.subplot(1, 2, 2)
            plt.hist(data[:, 1],
                     density=True,
                     bins=data.shape[0] // 10,
                     label=headers[1])
            plt.hist(data[:, 3],
                     density=True,
                     bins=data.shape[0] // 10,
                     label=headers[3])
            plt.legend()
            plt.suptitle("{}_{}_{}_{}_{}_{}_{}_{}".format(
                scene[:-2], scene[-1:], str(idx_frame),
                "mc" if motion_constraint else "!mc", experiment_group,
                str(res[0]) + 'x' +
                str(res[1]) if experiment_group == "noise" else noise, pts,
                opt_version))
            plt.show()
    elif experiment_group == "fov":
        noise = noises[0]
        for res in ress:
            dt = pd.read_csv(
                "../report/{}/{}/{}/{}/{}/{}/{}_{}_{}_{}_{}_{}_{}_{}.csv".
                format(scene, str(idx_frame),
                       "mc" if motion_constraint else "!mc", noise,
                       str(res[0]) + 'x' + str(res[1]), pts, scene[:-2],
                       scene[-1:], str(idx_frame),
                       "mc" if motion_constraint else "!mc", noise,
                       str(res[0]) + 'x' + str(res[1]), pts, opt_version))

            headers = dt.columns.values
            data = dt.values

            plt.figure()
            plt.subplot(1, 2, 1)
            plt.hist(data[:, 0],
                     density=True,
                     bins=data.shape[0] // 10,
                     label=headers[0])
            plt.hist(data[:, 2],
                     density=True,
                     bins=data.shape[0] // 10,
                     label=headers[2])
            plt.legend()
            plt.subplot(1, 2, 2)
            plt.hist(data[:, 1],
                     density=True,
                     bins=data.shape[0] // 10,
                     label=headers[1])
            plt.hist(data[:, 3],
                     density=True,
                     bins=data.shape[0] // 10,
                     label=headers[3])
            plt.legend()
            plt.suptitle("{}_{}_{}_{}_{}_{}_{}_{}".format(
                scene[:-2], scene[-1:], str(idx_frame),
                "mc" if motion_constraint else "!mc", experiment_group,
                str(res[0]) + 'x' +
                str(res[1]) if experiment_group == "noise" else noise, pts,
                opt_version))
            plt.show()


def plot_median_errors():
    _ours = []
    _8pa = []

    if experiment_group == "noise":
        res = ress[1]
        for noise in reversed(noises):
            dt = pd.read_csv(
                "../report/{}/{}/{}/{}/{}/{}/{}_{}_{}_{}_{}_{}_{}_{}.csv".
                format(scene, str(idx_frame),
                       "mc" if motion_constraint else "!mc", noise,
                       str(res[0]) + 'x' + str(res[1]), pts, scene[:-2],
                       scene[-1:], str(idx_frame),
                       "mc" if motion_constraint else "!mc", noise,
                       str(res[0]) + 'x' + str(res[1]), pts, opt_version))
            data = dt.values
            # ! Ours' method
            _ours.append(np.median(data[:, 2:4], axis=0))
            # ! 8PA
            _8pa.append(np.median(data[:, 0:2], axis=0))

    elif experiment_group == "fov":
        noise = noises[0]
        for res in ress:
            dt = pd.read_csv(
                "../report/{}/{}/{}/{}/{}/{}/{}_{}_{}_{}_{}_{}_{}_{}.csv".
                format(scene, str(idx_frame),
                       "mc" if motion_constraint else "!mc", noise,
                       str(res[0]) + 'x' + str(res[1]), pts, scene[:-2],
                       scene[-1:], str(idx_frame),
                       "mc" if motion_constraint else "!mc", noise,
                       str(res[0]) + 'x' + str(res[1]), pts, opt_version))
            data = dt.values
            # ! Ours' method
            _ours.append(np.median(data[:, 2:4], axis=0))
            # ! 8PA
            _8pa.append(np.median(data[:, 0:2], axis=0))

    print(_ours)
    print(_8pa)

    _ours = np.array(_ours)
    _8pa = np.array(_8pa)

    plt.figure(figsize=(4.5, 3))

    plt.plot(_8pa[:, 0], label='8pa-rot')
    plt.plot(_8pa[:, 1], label='8pa-trans')
    plt.plot(_ours[:, 0], label='ours-rot')
    plt.plot(_ours[:, 1], label='ours-trans')
    plt.legend(loc='upper left')

    x = [i for i in range(len(ress))]
    labels = reversed(degs) if experiment_group == "noise" else ress
    plt.xticks(x, labels, rotation='horizontal')
    plt.xlabel(experiment_group[0].upper() + experiment_group[1:])
    plt.ylabel('Error')
    plt.grid()

    plt.title("{}_{}_{}_{}_{}_{}_{}_{}".format(
        scene[:-2], scene[-1:], str(idx_frame),
        "mc" if motion_constraint else "!mc", experiment_group,
        str(res[0]) + 'x' +
        str(res[1]) if experiment_group == "noise" else noise, pts,
        opt_version))
    plt.subplots_adjust(left=0.175, right=0.885, bottom=0.160, top=0.895)
    plt.savefig("../report/{}/{}/{}_{}_{}_{}_{}_{}_{}_{}.png".format(
        scene, str(idx_frame), scene[:-2], scene[-1:], str(idx_frame),
        "mc" if motion_constraint else "!mc", experiment_group,
        str(res[0]) + 'x' +
        str(res[1]) if experiment_group == "noise" else noise, pts,
        opt_version))
    plt.show()


if __name__ == "__main__":
    # plot_histograms_of_errors()
    plot_median_errors()
