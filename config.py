# path = "/home/kike/Documents/datasets/Matterport_360_odometry"
path = "/run/user/1001/gvfs/sftp:host=140.114.27.95,port=50002/NFS/kike/minos/vslab_MP3D_VO/512x1024"
scene = "1LXtFkjw3qL" + "/0"
idx_frame = 0

opt_version = "v2"
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

# "noise", "fov", "point"
experiment_group = "point"

noise = noises[0] if experiment_group != "noise" else None
res = ress[1] if experiment_group != "fov" else None
point = points[1] if experiment_group != "point" else None
