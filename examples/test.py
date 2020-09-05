import numpy as np
from pcl_utilities import *
from geometry_utilities import *

W = np.random.random((3, 3)) * 20

pcl_a = np.array([[1 for i in range(30)] * 4], dtype='f')
pcl_a = pcl_a.reshape(4, 30)
pcl_a[0:3, :] = np.array(np.random.random((3, 30)) * 10)

# plot_color_plc(pcl_a[0:3, :].T)
cam_gt = get_homogeneous_transform_from_vectors(r_vector=(0, 0, 0), t_vector=(1, 0, 0))

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
