# path = "/home/kike/Documents/datasets/Matterport_360_odometry"

# ! Choices: "minos", "tum_rgbd", "tum_vi", "kitti", "carla"
dataset = "minos"

if dataset == "minos":
    path = "/run/user/1001/gvfs/sftp:host=140.114.27.95,port=50002/NFS/kike/minos/3dv2020"
    scene = "1LXtFkjw3qL" + "/1"
# elif dataset == "tum_rgbd":
#     path = "/run/user/1001/gvfs/sftp:host=140.114.27.95,port=50002/NFS/kike/minos/3dv2020"
#     scene = "1LXtFkjw3qL" + "/1"

idx_frame = 230

# ! Choices: "v0", "v1", "v2", "v2.1"
opt_version = "v2.1"
motion_constraint = True if opt_version == "v0" else False

# ! FoV
ress = [(54.4, 37.8), (65.5, 46.4), (195, 195), (360, 180)]

# ! Noise
noises = [500, 1000, 2000, 10000]
# noises = [i for i in range(100, 10000 + 1, 100)]

# ! Point
points = [8, 150, 500, 1000]
# points = [i for i in range(10, 1000 + 1, 10)]
degs = [3.21, 2.27, 1.60, 0.72]

# ! Choices: "noise", "fov", "point"
experiment_group = "noise"

noise = noises[0] if experiment_group != "noise" else None
res = ress[1] if experiment_group != "fov" else None
point = points[1] if experiment_group != "point" else None
