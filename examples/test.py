import numpy as np
from pcl_utilities import *
from geometry_utilities import *

W = np.random.random((3, 3)) * 20

pcl_a = np.array([[1 for i in range(30)] * 4], dtype='f')
pcl_a = pcl_a.reshape(4, 30)
pcl_a[0:3, :] = np.array(np.random.random((3, 30)) * 10)

# plot_color_plc(pcl_a[0:3, :].T)

R = np.random.random((3, 3))
t = np.random.random((3, 1))
cam_gt = np.eye(4)
cam_gt[0:3, 0:3] = R
cam_gt[0:3, 3] = np.reshape(t, (3))

pcl_b = np.linalg.inv(cam_gt).dot(pcl_a)

# plot_color_plc(pcl_b[0:3, :].T)

from solvers.pnp import PnP
pnp = PnP()
bearings_b = sph.sphere_normalization(pcl_b)
cam_pnp = pnp.recoverPose(
            w=pcl_a.copy(),
            x=bearings_b.copy())

err = evaluate_error_in_transformation(transform_gt=cam_gt,
                                       transform_est=cam_pnp)

print(err)
print("done")
