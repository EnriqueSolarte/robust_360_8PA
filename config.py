# path = "/home/kike/Documents/datasets/Matterport_360_odometry"

# ! Choices: "minos", "tum_rgbd", "tum_vi", "kitti", "carla"
dataset = "minos"

# ! Output directory
output_dir = "/home/justin/slam/optimal8PA/report"

if dataset == "minos":
    # path = "/home/justin/slam/openvslam_norm/python_scripts/synthetic_points_exp/data/3dv2020"
    # path = "/run/user/1001/gvfs/sftp:host=140.114.27.95,port=50002/NFS/kike/minos/3dv2020"
    path = "/home/kike/Documents/datasets/MP3D_VO"
    scene = "2azQ1b91cZZ" + "/0"
    idx_frame = 549
elif dataset == "tum_rgbd":
    path = "/run/user/1001/gvfs/sftp:host=140.114.27.95,port=50002/NFS/kike/undistort_depth/Testing_and_debugging/unzip"
    scene = "rgbd_dataset_freiburg1_xyz"

# ! Choices: "v0", "v1", "v2", "v2.1"
opt_version_choices = ["v0", "v1", "v2", "v2.1"]
opt_version = opt_version_choices[1]
motion_constraint = True if opt_version == "v0" else False

# ! FoV
ress = [(54.4, 37.8), (65.5, 46.4), (180, 180), (360, 180)]

# ! Noise
noises = [500, 1000, 2000, 10000]
# noises = [500]
# noises = [i for i in range(100, 10000 + 1, 500)]

# ! Point
points = [8, 150, 500, 1000]
# points = list(range(8, 101, 10))
# points = [i for i in range(10, 1000 + 1, 10)]
# degs = [3.21, 2.27, 1.60, 0.72]

# ! Choices: "noise", "fov", "point"
experiment_group_choices = ["noise", "fov", "point", "frame"]
experiment_group = experiment_group_choices[3]

noise = noises[0] if experiment_group != "noise" else None
res = ress[1] if experiment_group != "fov" else None
point = points[1] if experiment_group != "point" else None

experiment_choices = ["sample", "feature"]
experiment = experiment_choices[1]
