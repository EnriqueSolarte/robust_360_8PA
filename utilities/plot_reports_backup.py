import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def plot_histograms_of_errors(scene,
                              idx_frame,
                              opt_version,
                              motion_constraint,
                              r=True):
    ress = [(54.4, 37.8), (65.5, 46.4), (195, 195), (360, 180)]

    if r:
        for res in ress:
            dt = pd.read_csv(
                "../report/{}/{}/{}/{}_{}x{}.csv".format(
                    scene, str(idx_frame), "mc"
                    if motion_constraint else "!mc", opt_version, str(res[0]),
                    str(res[1])),
                delimiter=" ")
            headers = dt.columns.values
            data = dt.values
            print(
                "====================================================================="
            )
            # ! Ours' method
            print("Q1-ours:{}- {}".format(
                np.quantile(data[:, 2:4], 0.25, axis=0), len(data[:, 0])))
            print("Q2-ours:{}- {}".format(
                np.median(data[:, 2:4], axis=0), len(data[:, 0])))
            print("Q3-ours:{}- {}".format(
                np.quantile(data[:, 2:4], 0.75, axis=0), len(data[:, 0])))
            print(
                "====================================================================="
            )
            # ! 8PA
            print("Q1-8PA:{}-  {}".format(
                np.quantile(data[:, 0:2], 0.25, axis=0), len(data[:, 0])))
            print("Q2-8PA:{}-  {}".format(
                np.median(data[:, 0:2], axis=0), len(data[:, 0])))
            print("Q3-8PA:{}-  {}".format(
                np.quantile(data[:, 0:2], 0.75, axis=0), len(data[:, 0])))
            print(
                "====================================================================="
            )
            plt.figure()
            plt.subplot(1, 2, 1)
            plt.hist(
                data[:, 0],
                density=True,
                bins=data.shape[0] // 10,
                label=headers[0])
            plt.hist(
                data[:, 2],
                density=True,
                bins=data.shape[0] // 10,
                label=headers[2])
            plt.legend()
            plt.subplot(1, 2, 2)
            plt.hist(
                data[:, 1],
                density=True,
                bins=data.shape[0] // 10,
                label=headers[1])
            plt.hist(
                data[:, 3],
                density=True,
                bins=data.shape[0] // 10,
                label=headers[3])
            plt.legend()
            plt.title("{}-{}-{}-{}-{}x{}".format(scene, str(idx_frame), "mc"
                                                 if motion_constraint
                                                 else "!mc", opt_version,
                                                 str(res[0]), str(res[1])))
            plt.show()
    else:
        noises = [500, 1000, 2000, 10000]
        for noise in noises:
            dt = pd.read_csv(
                "../report/{}/{}/{}/{}/{}_{}x{}.csv".format(
                    scene, str(idx_frame), "mc"
                    if motion_constraint else "!mc", noise, opt_version,
                    str(ress[1][0]), str(ress[1][1])),
                delimiter=" ")
            headers = dt.columns.values
            data = dt.values
            print(
                "====================================================================="
            )
            # ! Ours' method
            print("Q1-ours:{}- {}".format(
                np.quantile(data[:, 2:4], 0.25, axis=0), len(data[:, 0])))
            print("Q2-ours:{}- {}".format(
                np.median(data[:, 2:4], axis=0), len(data[:, 0])))
            print("Q3-ours:{}- {}".format(
                np.quantile(data[:, 2:4], 0.75, axis=0), len(data[:, 0])))
            print(
                "====================================================================="
            )
            # ! 8PA
            print("Q1-8PA:{}-  {}".format(
                np.quantile(data[:, 0:2], 0.25, axis=0), len(data[:, 0])))
            print("Q2-8PA:{}-  {}".format(
                np.median(data[:, 0:2], axis=0), len(data[:, 0])))
            print("Q3-8PA:{}-  {}".format(
                np.quantile(data[:, 0:2], 0.75, axis=0), len(data[:, 0])))
            print(
                "====================================================================="
            )
            plt.figure()
            plt.subplot(1, 2, 1)
            plt.hist(
                data[:, 0],
                density=True,
                bins=data.shape[0] // 10,
                label=headers[0])
            plt.hist(
                data[:, 2],
                density=True,
                bins=data.shape[0] // 10,
                label=headers[2])
            plt.legend()
            plt.subplot(1, 2, 2)
            plt.hist(
                data[:, 1],
                density=True,
                bins=data.shape[0] // 10,
                label=headers[1])
            plt.hist(
                data[:, 3],
                density=True,
                bins=data.shape[0] // 10,
                label=headers[3])
            plt.legend()
            plt.title("{}-{}-{}-{}-{}-{}x{}".format(scene, str(idx_frame), "mc"
                                                    if motion_constraint else
                                                    "!mc", noise, opt_version,
                                                    str(ress[1][0]),
                                                    str(ress[1][1])))
            plt.show()


def plot_frame_errors(filename):
    dt = pd.read_csv(filename, delimiter=" ")
    headers = dt.columns.values
    data = dt.values
    print(
        "====================================================================="
    )
    # ! Ours' method
    print("Q1-ours:{}- {}".format(
        np.quantile(data[:, 2:4], 0.25, axis=0), len(data[:, 0])))
    print("Q2-ours:{}- {}".format(
        np.median(data[:, 2:4], axis=0), len(data[:, 0])))
    print("Q3-ours:{}- {}".format(
        np.quantile(data[:, 2:4], 0.75, axis=0), len(data[:, 0])))
    print(
        "====================================================================="
    )
    # ! 8PA
    print("Q1-8PA:{}-  {}".format(
        np.quantile(data[:, 0:2], 0.25, axis=0), len(data[:, 0])))
    print("Q2-8PA:{}-  {}".format(
        np.median(data[:, 0:2], axis=0), len(data[:, 0])))
    print("Q3-8PA:{}-  {}".format(
        np.quantile(data[:, 0:2], 0.75, axis=0), len(data[:, 0])))
    print(
        "====================================================================="
    )
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.plot(data[:, 0], label=headers[0])
    plt.plot(data[:, 2], label=headers[2])
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(data[:, 1], label=headers[1])
    plt.plot(data[:, 3], label=headers[3])
    plt.legend()
    plt.show()


def plot_median_errors(scene,
                       idx_frame,
                       opt_version,
                       motion_constraint,
                       r=True):
    ress = [(54.4, 37.8), (65.5, 46.4), (195, 195), (360, 180)]
    # ress = [(3.44, 5.15), (16.1, 23.9), (27.0, 39.6), (54.4, 37.8), (65.5, 46.4), (81.2, 102.7), (195, 195), (360, 180)]

    _ours = []
    _8pa = []

    if r:
        for res in ress:
            dt = pd.read_csv(
                "../report/{}/{}/{}/{}_{}x{}.csv".format(
                    scene, str(idx_frame), "mc"
                    if motion_constraint else "!mc", opt_version, str(res[0]),
                    str(res[1])),
                delimiter=" ")
            headers = dt.columns.values
            data = dt.values

            # ! Ours' method
            _ours.append(np.median(data[:, 2:4], axis=0))
            # ! 8PA
            _8pa.append(np.median(data[:, 0:2], axis=0))

        print(_ours)
        print(_8pa)

        _ours = np.array(_ours)
        _8pa = np.array(_8pa)

        plt.figure(figsize=(5.2, 2.34))

        plt.plot(_8pa[:, 0], label='8pa-rot')
        plt.plot(_8pa[:, 1], label='8pa-trans')
        plt.plot(_ours[:, 0], label='ours-rot')
        plt.plot(_ours[:, 1], label='ours-trans')

        x = [i for i in range(len(ress))]
        labels = ['({}x{})'.format(str(res[0]), str(res[1])) for res in ress]
        plt.xticks(x, labels, rotation='horizontal')
        plt.xlabel('FoV')
        plt.ylabel('Error')
        plt.grid()

        # plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
        plt.title("{}-{}-{}-{}-{}".format(scene, str(idx_frame), "mc"
                                          if motion_constraint else "!mc",
                                          opt_version, 500))
        plt.subplots_adjust(left=0.125, right=0.955, bottom=0.230, top=0.865)
        plt.savefig("../report/{}/{}/{}/{}_{}_fov.png".format(
            scene, str(idx_frame), "mc"
            if motion_constraint else "!mc", 500, opt_version, str(ress[1][0]),
            str(ress[1][1])))
        plt.show()
    else:
        noises = [500, 1000, 2000, 10000]
        degs = [3.21, 2.27, 1.60, 0.72]

        for noise in reversed(noises):
            dt = pd.read_csv(
                "../report/{}/{}/{}/{}/{}_{}x{}.csv".format(
                    scene, str(idx_frame), "mc"
                    if motion_constraint else "!mc", noise, opt_version,
                    str(ress[1][0]), str(ress[1][1])),
                delimiter=" ")
            headers = dt.columns.values
            data = dt.values

            # ! Ours' method
            _ours.append(np.median(data[:, 2:4], axis=0))
            # ! 8PA
            _8pa.append(np.median(data[:, 0:2], axis=0))

        print(_ours)
        print(_8pa)

        _ours = np.array(_ours)
        _8pa = np.array(_8pa)

        plt.figure(figsize=(3.13, 2.34))

        plt.plot(_8pa[:, 0], label='8pa-rot')
        plt.plot(_8pa[:, 1], label='8pa-trans')
        plt.plot(_ours[:, 0], label='ours-rot')
        plt.plot(_ours[:, 1], label='ours-trans')

        x = [i for i in range(len(ress))]
        labels = reversed(degs)
        plt.xticks(x, labels, rotation='horizontal')
        plt.xlabel('Noise')
        plt.ylabel('Error')
        plt.grid()

        # plt.legend(loc="upper left")
        plt.title("{}-{}-{}-{}-{}x{}".format(scene, str(idx_frame), "mc"
                                             if motion_constraint else "!mc",
                                             opt_version, str(ress[1][0]),
                                             str(ress[1][1])))
        plt.subplots_adjust(left=0.215, right=0.780, bottom=0.195, top=0.885)
        plt.savefig("../report/{}/{}/{}/{}_{}x{}_noise.png".format(
            scene, str(idx_frame), "mc"
            if motion_constraint else "!mc", opt_version, str(ress[1][0]),
            str(ress[1][1])))
        plt.show()


if __name__ == '__main__':
    # plot_histograms_of_errors("../report/1LXtFkjw3qL/0/mc/v1_54.4x37.8.csv")
    # plot_histograms_of_errors("../report/1LXtFkjw3qL/0/mc/v1_65.5x46.4.csv")
    # plot_histograms_of_errors("../report/1LXtFkjw3qL/0/mc/v1_195x195.csv")
    # plot_histograms_of_errors("../report/1LXtFkjw3qL/0/mc/v1_360x180.csv")

    # plot_histograms_of_errors("../report/v2_1LXtFkjw3qL_195x195_149.csv")
    # plot_histograms_of_errors("../report/v2_1LXtFkjw3qL_360x180_149.csv")

    # plot_histograms_of_errors(motion_constraint=True, opt_version="v0", scene="zsNo4HB9uLZ", idx_frame=40, r=False)
    plot_median_errors(
        motion_constraint=True,
        opt_version="v2",
        scene="1LXtFkjw3qL",
        idx_frame=0,
        r=False)

    # plot_median_errors(motion_constraint=True, opt_version="v2", scene="1LXtFkjw3qL", idx_frame=0)

    # plot_histograms_of_errors("../report/v1_sample_scene.csv", fig=0)
    # plot_histograms_of_errors("../report/v2_sample_scene.csv", fig=1)
    # plot_histograms_of_errors("../report/v1_sample_scene.csv", fig=1)
    # plot_frame_errors("../report/v1_sequence_frames.csv", fig=2)
    # plot_frame_errors("../report/v2_sequence_frames.csv", fig=3)
