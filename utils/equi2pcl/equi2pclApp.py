from Reading_from_datasets.reading_MP3D_VO import Data
from projections.equi2pcl.equi2pcl import Equi2PCL
import numpy as np
from geometry_utilities import extend_array_to_homogeneous
from vispy_utilities import plot_color_plc

# ! this data is in data/sample_images
scene_ = "1LXtFkjw3qL"
path_ = "0"
# ! NAS-NFS SERVER --> in kike/minos/vslab_MP3D_VO
root_ = '/run/user/1000/gvfs/sftp:host=127.0.0.1,port=8002/NFS/kike/minos/vslab_MP3D_VO'
# root_ = "/run/user/1000/gvfs/sftp:host=140.114.27.95,port=50002/NFS/kike/minos/vslab_MP3D_VO"
dt = Data(scene=scene_, basedir=path_, root_data=root_)
# dt.set_png_for_depth()
e2pcl = Equi2PCL(shape=dt.shape)

# ! Frame index: 0
color_map, depth_map, pose = dt.get_data_frame(0)
pcl, color_pcl = e2pcl.get_pcl(color_map=color_map, depth_map=depth_map)
pcl = extend_array_to_homogeneous(pcl)

mask = pcl[1, :] > 0
pcl_w = pcl[:, mask]
color_w = color_pcl[:, mask]

# ! Reading the last frame
color_map, depth_map, pose = dt.get_data_frame(dt.number_frames - 1)
pcl, color_pcl = e2pcl.get_pcl(color_map=color_map, depth_map=depth_map)
pcl = extend_array_to_homogeneous(pcl)
mask = pcl[1, :] > 0

pcl_w = np.append(pcl_w, pose.dot(pcl[:, mask]), 1)
color_w = np.append(color_w, color_pcl[:, mask], 1)

plot_color_plc(pcl_w[0:3, :].T, color_w.T, background="white")
