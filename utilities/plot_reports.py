import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def plot_histograms_of_errors(filename, fig=0):
    dt = pd.read_csv(filename, delimiter=" ")
    headers = dt.columns.values
    data = dt.values
    plt.figure(fig)
    plt.subplot(1, 2, 1)
    plt.hist(data[abs(data[:, 1]) < 30, 1], density=True, bins=data.shape[0] // 10, label=headers[0])
    plt.hist(data[abs(data[:, 2]) < 30, 2], density=True, bins=data.shape[0] // 10, label=headers[2])
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.hist(data[abs(data[:, 1]) < 30, 1], density=True, bins=data.shape[0] // 10, label=headers[1])
    plt.hist(data[abs(data[:, 3]) < 30, 3], density=True, bins=data.shape[0] // 10, label=headers[3])
    plt.legend()
    plt.show()


def plot_frame_errors(filename, fig=0):
    dt = pd.read_csv(filename, delimiter=" ")
    headers = dt.columns.values
    data = dt.values
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
