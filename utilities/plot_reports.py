import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from config import *


def plot_histograms_of_errors(noise, res, point):
    if experiment_group == "noise":
        for noise in reversed(noises):
            dt = pd.read_csv(
                "../report/{}/{}/{}/{}/{}/{}/{}_{}_{}_{}_{}_{}_{}_{}.csv".
                format(scene, str(idx_frame),
                       "mc" if motion_constraint else "!mc", noise,
                       str(res[0]) + "x" + str(res[1]), point, scene[:-2],
                       scene[-1:], str(idx_frame),
                       "mc" if motion_constraint else "!mc", noise,
                       str(res[0]) + "x" + str(res[1]), point, opt_version))

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
            plt.suptitle("{}_{}_{}_{}_{}_{}_{}_{}_{}".format(
                scene[:-2], scene[-1:], str(idx_frame),
                "mc" if motion_constraint else "!mc", experiment_group,
                noise if experiment_group != "noise" else "",
                str(res[0]) + "x" +
                str(res[1]) if experiment_group != "fov" else "",
                point if experiment_group != "point" else "", opt_version))
            plt.show()
    elif experiment_group == "fov":
        for res in ress:
            dt = pd.read_csv(
                "../report/{}/{}/{}/{}/{}/{}/{}_{}_{}_{}_{}_{}_{}_{}.csv".
                format(scene, str(idx_frame),
                       "mc" if motion_constraint else "!mc", noise,
                       str(res[0]) + "x" + str(res[1]), point, scene[:-2],
                       scene[-1:], str(idx_frame),
                       "mc" if motion_constraint else "!mc", noise,
                       str(res[0]) + "x" + str(res[1]), point, opt_version))

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
            plt.suptitle("{}_{}_{}_{}_{}_{}_{}_{}_{}".format(
                scene[:-2], scene[-1:], str(idx_frame),
                "mc" if motion_constraint else "!mc", experiment_group,
                noise if experiment_group != "noise" else "",
                str(res[0]) + "x" +
                str(res[1]) if experiment_group != "fov" else "",
                point if experiment_group != "point" else "", opt_version))
            plt.show()
    elif experiment_group == "point":
        for point in points:
            dt = pd.read_csv(
                "../report/{}/{}/{}/{}/{}/{}/{}_{}_{}_{}_{}_{}_{}_{}.csv".
                format(scene, str(idx_frame),
                       "mc" if motion_constraint else "!mc", noise,
                       str(res[0]) + "x" + str(res[1]), point, scene[:-2],
                       scene[-1:], str(idx_frame),
                       "mc" if motion_constraint else "!mc", noise,
                       str(res[0]) + "x" + str(res[1]), point, opt_version))

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
            plt.suptitle("{}_{}_{}_{}_{}_{}_{}_{}_{}".format(
                scene[:-2], scene[-1:], str(idx_frame),
                "mc" if motion_constraint else "!mc", experiment_group,
                noise if experiment_group != "noise" else "",
                str(res[0]) + "x" +
                str(res[1]) if experiment_group != "fov" else "",
                point if experiment_group != "point" else "", opt_version))
            plt.show()


def plot_median_errors(noise, res, point):
    _ours = []
    _8pa = []

    if experiment_group == "noise":
        for noise in reversed(noises):
            dt = pd.read_csv(
                "../report/{}/{}/{}/{}/{}/{}/{}_{}_{}_{}_{}_{}_{}_{}.csv".
                format(scene, str(idx_frame),
                       "mc" if motion_constraint else "!mc", noise,
                       str(res[0]) + "x" + str(res[1]), point, scene[:-2],
                       scene[-1:], str(idx_frame),
                       "mc" if motion_constraint else "!mc", noise,
                       str(res[0]) + "x" + str(res[1]), point, opt_version))
            data = dt.values
            # ! Ours' method
            _ours.append(np.median(data[:, 2:4], axis=0))
            # ! 8PA
            _8pa.append(np.median(data[:, 0:2], axis=0))
    elif experiment_group == "fov":
        for res in ress:
            dt = pd.read_csv(
                "../report/{}/{}/{}/{}/{}/{}/{}_{}_{}_{}_{}_{}_{}_{}.csv".
                format(scene, str(idx_frame),
                       "mc" if motion_constraint else "!mc", noise,
                       str(res[0]) + "x" + str(res[1]), point, scene[:-2],
                       scene[-1:], str(idx_frame),
                       "mc" if motion_constraint else "!mc", noise,
                       str(res[0]) + "x" + str(res[1]), point, opt_version))
            data = dt.values
            # ! Ours' method
            _ours.append(np.median(data[:, 2:4], axis=0))
            # ! 8PA
            _8pa.append(np.median(data[:, 0:2], axis=0))
    elif experiment_group == "point":
        for point in points:
            dt = pd.read_csv(
                "../report/{}/{}/{}/{}/{}/{}/{}_{}_{}_{}_{}_{}_{}_{}.csv".
                format(scene, str(idx_frame),
                       "mc" if motion_constraint else "!mc", noise,
                       str(res[0]) + "x" + str(res[1]), point, scene[:-2],
                       scene[-1:], str(idx_frame),
                       "mc" if motion_constraint else "!mc", noise,
                       str(res[0]) + "x" + str(res[1]), point, opt_version))
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
    # plt.legend(loc='upper left')

    if experiment_group == "noise":
        x = [i for i in range(len(noises))]
        # labels = reversed(degs)
        labels = reversed(noises)

        # x = list(range(0, len(noises) + 1, int(len(noises) / 4)))
        # labels = list(range(0, noises[-1] + 1, int(noises[-1] / 4)))
        # labels = reversed(labels)
    elif experiment_group == "fov":
        x = [i for i in range(len(ress))]
        labels = ress

        # x = list(range(0, len(ress) + 1, int(len(ress) / 4)))
        # labels = list(range(0, ress[-1] + 1, int(ress[-1] / 4)))
    elif experiment_group == "point":
        x = [i for i in range(len(points))]
        labels = points

        # x = list(range(0, len(points) + 1, int(len(points) / 4)))
        # labels = list(range(0, points[-1] + 1, int(points[-1] / 4)))

    plt.xticks(x, labels, rotation='horizontal')
    plt.xlabel(experiment_group[0].upper() + experiment_group[1:])
    plt.ylabel('Error')
    plt.grid()

    # plt.title("{}_{}_{}_{}_{}_{}_{}_{}".format(
    #     scene[:-2], scene[-1:], str(idx_frame),
    #     "mc" if motion_constraint else "!mc", experiment_group,
    #     str(res[0]) + "x" +
    #     str(res[1]) if experiment_group == "noise" else noise, point,
    #     opt_version))

    plt.suptitle("{}_{}_{}_{}_{}_{}_{}_{}_{}".format(
        scene[:-2], scene[-1:], str(idx_frame),
        "mc" if motion_constraint else "!mc", experiment_group,
        noise if experiment_group != "noise" else "",
        str(res[0]) + "x" + str(res[1]) if experiment_group != "fov" else "",
        point if experiment_group != "point" else "", opt_version))

    plt.subplots_adjust(left=0.175, right=0.885, bottom=0.160, top=0.895)
    plt.savefig("../report/{}/{}/{}_{}_{}_{}_{}_{}_{}_{}_{}.png".format(
        scene, str(idx_frame), scene[:-2], scene[-1:], str(idx_frame),
        "mc" if motion_constraint else "!mc", experiment_group,
        noise if experiment_group != "noise" else "",
        str(res[0]) + "x" + str(res[1]) if experiment_group != "fov" else "",
        point if experiment_group != "point" else "", opt_version))
    plt.show()


if __name__ == "__main__":
    # plot_histograms_of_errors()
    plot_median_errors(noise, res, point)
