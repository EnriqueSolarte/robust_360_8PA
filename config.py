# path = "/home/kike/Documents/datasets/Matterport_360_odometry"
path = "/run/user/1001/gvfs/sftp:host=140.114.27.95,port=50002/NFS/kike/minos/vslab_MP3D_VO/512x1024"
scene = "1LXtFkjw3qL" + "/0"
idx_frame = 0
pts = 150

opt_version = "v2"
if opt_version == "v0":
    motion_constraint = True
else:
    motion_constraint = False

ress = [(54.4, 37.8), (65.5, 46.4), (195, 195), (360, 180)]
noises = [500, 1000, 2000, 10000]
degs = [3.21, 2.27, 1.60, 0.72]

experiment_group = "noise"

res = ress[1] if experiment_group == "noise" else None
noise = noises[0] if experiment_group == "fov" else None
