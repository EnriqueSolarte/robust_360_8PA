import os
from utils.TUM_RGBD_utils.associate import associate, read_file_list
from utils.TUM_RGBD_utils.evaluate_rpe import read_trajectory
import pandas as pd


def save_association(caption, file_association, data):
    label = list(data.keys())
    dt = pd.DataFrame.from_dict(data)
    dt.to_csv(file_association, header=False, index=False, sep=" ")

    header = caption + "\n#" + str(label)
    with open(file_association, 'r+') as f:
        content = f.read()
        f.seek(0, 0)
        f.write(header.rstrip('\r\n') + '\n' + content)

    print('Done')


def create_associations(basedir, scene, **kwargs):
    print("creating association files !!!")
    # basedir = "/home/kike/Documents/datasets/TUM_RGBD"
    # scene = "Handheld/rgbd_dataset_freiburg1_360"
    # scene = "Handheld/rgbd_dataset_freiburg2_large_with_loop"
    # scene = "Handheld/rgbd_dataset_freiburg3_long_office_household"

    scene_dir = os.path.join(basedir, scene)
    # *  reading dataset files
    depth_image_paths = os.path.join(scene_dir, 'depth.txt')
    rgb_image_paths = os.path.join(scene_dir, 'rgb.txt')
    trajectory_gt = os.path.join(scene_dir, 'groundtruth.txt')

    # !initial dictionaries
    dic_rgb = read_file_list(rgb_image_paths)
    dic_depth = read_file_list(depth_image_paths)
    dic_pose = read_trajectory(trajectory_gt)

    delta_pose2depth = delta["delta_pose2depth"]
    delta_rgb2depth = delta["delta_rgb2depth"]
    delta_pose2rgb = delta["delta_pose2rgb"]

    # ! Ours dictionaries
    dic_pose2depth = dict(associate(dic_pose, dic_depth, 0, delta_pose2depth))

    dic_pose2rgb = dict(associate(dic_pose, dic_rgb, 0, delta_pose2rgb))

    dic_timeRGB2timeDepth = dict(
        associate(dic_pose2rgb, dic_pose2depth, 0, delta_rgb2depth))

    caption = "# File association for accurate rgb images with cam pose (delta_pose2rgb={})".format(
        delta_pose2rgb)
    file_association = os.path.join(scene_dir, "pose_rgb_association.txt")
    data = dict(
        time_pose=sorted(list(dic_pose2rgb.keys())),
        time_rgb=[dic_pose2rgb[t] for t in sorted(list(dic_pose2rgb.keys()))])
    save_association(caption, file_association, data)

    caption = "# File association for accurate depth images with cam pose (delta_pose2depth={})".format(
        delta_pose2depth)
    file_association = os.path.join(scene_dir, "pose_depth_association.txt")
    data = dict(time_pose=sorted(list(dic_pose2depth.keys())),
                time_depth=[
                    dic_pose2depth[t]
                    for t in sorted(list(dic_pose2depth.keys()))
                ])
    save_association(caption, file_association, data)

    caption = "# File association for accurate depth images with cam pose (" + \
              "delta_pose2rgb={}, ".format(delta_pose2rgb) + \
              "delta_pose2depth={}, ".format(delta_pose2depth) + \
              "delta_rgb2depth={})".format(delta_rgb2depth)

    file_association = os.path.join(scene_dir,
                                    "pose_depth_rgb_association.txt")
    time_pose = sorted(list(dic_timeRGB2timeDepth.keys()))
    time_pose2depth = [dic_timeRGB2timeDepth[t] for t in time_pose]
    data = dict(time_pose=time_pose,
                time_depth=[dic_pose2depth[t] for t in time_pose2depth],
                time_rgb=[dic_pose2rgb[t] for t in time_pose])
    save_association(caption, file_association, data)
