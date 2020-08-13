import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def plot_histograms_of_errors(filename, fig=0):
    dt = pd.read_csv(filename, delimiter=" ")
    headers = dt.columns.values
    data = dt.values
    print("=====================================================================")
    # ! Ours' method
    print("Q1-ours:{}- {}".format(np.quantile(data[:, 2:4], 0.25, axis=0), len(data[:, 0])))
    print("Q2-ours:{}- {}".format(np.median(data[:, 2:4], axis=0), len(data[:, 0])))
    print("Q3-ours:{}- {}".format(np.quantile(data[:, 2:4], 0.75, axis=0), len(data[:, 0])))
    print("=====================================================================")
    # ! 8PA
    print("Q1-8PA:{}-  {}".format(np.quantile(data[:, 0:2], 0.25, axis=0), len(data[:, 0])))
    print("Q2-8PA:{}-  {}".format(np.median(data[:, 0:2], axis=0), len(data[:, 0])))
    print("Q3-8PA:{}-  {}".format(np.quantile(data[:, 0:2], 0.75, axis=0), len(data[:, 0])))
    print("=====================================================================")
    plt.figure(fig)
    plt.subplot(1, 2, 1)
    plt.hist(data[:, 1], density=True, bins=data.shape[0] // 10, label=headers[0])
    plt.hist(data[:, 2], density=True, bins=data.shape[0] // 10, label=headers[2])
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.hist(data[:, 1], density=True, bins=data.shape[0] // 10, label=headers[1])
    plt.hist(data[:, 3], density=True, bins=data.shape[0] // 10, label=headers[3])
    plt.legend()
    plt.show()


def plot_frame_errors(filename, fig=0):
    dt = pd.read_csv(filename, delimiter=" ")
    headers = dt.columns.values
    data = dt.values
    print("=====================================================================")
    # ! Ours' method
    print("Q1-ours:{}- {}".format(np.quantile(data[:, 2:4], 0.25, axis=0), len(data[:, 0])))
    print("Q2-ours:{}- {}".format(np.median(data[:, 2:4], axis=0), len(data[:, 0])))
    print("Q3-ours:{}- {}".format(np.quantile(data[:, 2:4], 0.75, axis=0), len(data[:, 0])))
    print("=====================================================================")
    # ! 8PA
    print("Q1-8PA:{}-  {}".format(np.quantile(data[:, 0:2], 0.25, axis=0), len(data[:, 0])))
    print("Q2-8PA:{}-  {}".format(np.median(data[:, 0:2], axis=0), len(data[:, 0])))
    print("Q3-8PA:{}-  {}".format(np.quantile(data[:, 0:2], 0.75, axis=0), len(data[:, 0])))
    print("=====================================================================")
    plt.figure(fig)
    plt.subplot(1, 2, 1)
    plt.plot(data[:, 0], label=headers[0])
    plt.plot(data[:, 2], label=headers[2])
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(data[:, 1], label=headers[1])
    plt.plot(data[:, 3], label=headers[3])
    plt.legend()
    plt.show()


if __name__ == '__main__':
    plot_histograms_of_errors("../report/v1_sample_scene.scv", fig=0)
    plot_histograms_of_errors("../report/v2_sample_scene.scv", fig=1)
    # plot_histograms_of_errors("../report/v1_sample_scene.scv", fig=1)
    # plot_frame_errors("../report/v1_sequence_frames.scv", fig=2)
    # plot_frame_errors("../report/v2_sequence_frames.scv", fig=3)
