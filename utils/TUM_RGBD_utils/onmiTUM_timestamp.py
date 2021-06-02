from file_utilities import read_txt_file, write_report
import os


def get_column(path, index=None):
    data = read_txt_file(path, split_list="\t")
    if index is None:
        return data
    column = [dt[index] for dt in data]
    return column


if __name__ == '__main__':
    """
    This scripts fix the timestamp in TUM_Omni dataset
    The main objective is tho get a groundtruth.txt file according with the TUM-RGDB dataset
    In order to use the same API
    WARNING: This script is only for TUM_OMNI Dataset
    """

    main_path = '/home/kike/Documents/Dataset/TUM_Omnidirectional'

    scene = 'T2'
    version = 'orig'

    gt_file_orginal = main_path + '/{}/{}_{}/{}-GT.txt'.format(
        scene, scene, version, scene)
    images_file = main_path + '/{}/{}_{}/images.txt'.format(
        scene, scene, version, scene)

    timestamp_gt = get_column(gt_file_orginal)
    timestamp_images = get_column(images_file, 0)
    """Creating groundtruth.txt file"""
    output_file = '/home/kike/Documents/Dataset/TUM_Omnidirectional/{}/{}_{}/groundtruth.txt'.format(
        scene, scene, version)
    if os.path.isfile(output_file):
        os.remove(output_file)

    line = ["# ground truth trajectory"]
    write_report(output_file, line)
    line = ["# This file comes from: {}".format(gt_file_orginal)]
    write_report(output_file, line)
    line = ["# timestamp tx ty tz qx qy qz qw"]
    write_report(output_file, line)

    timestamp = float(timestamp_images[0])
    for timing, tx, ty, tz, _, _, _, _ in timestamp_gt:
        line = [
            str(timestamp + float(timing)) + " " + tx + " " + ty + " " + tz +
            " 0 0 0 0"
        ]
        timestamp = timestamp + float(timing)
        write_report(output_file, line)

    print("done")
