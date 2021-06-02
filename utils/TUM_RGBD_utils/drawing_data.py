from TUM_RGBD_utils.associate import read_file_list
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    file_data = '/home/kike/Documents/Dataset/TUM_Omnidirectional/T2/T2_rectified/T2-GT.txt'

    data_dic = read_file_list(file_data)
    data_list = list(data_dic)

    trajectory = []
    for stamp in data_list:
        data = data_dic[stamp]
        x = float(data[0])
        y = float(data[1])
        z = float(data[2])
        trajectory.append([x, y, z])

    trajectory = np.asanyarray(trajectory) / 1000
    fig = plt.figure()
    x = trajectory[:, 0]
    y = trajectory[:, 1]
    z = trajectory[:, 2]
    plt.scatter(x, y, z, c='g', marker='o')
    plt.show()
    print("done")
