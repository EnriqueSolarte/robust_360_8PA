from utils.image_utilities import *
from scipy.spatial import cKDTree
import math
from utils.camera_models import Camera


class Equirectangular(Camera):

    # region shape property
    @property
    def shape(self):
        return self.__shape

    @shape.setter
    def shape(self, value):
        assert isinstance(value, tuple)
        assert len(value) == 2
        self.__shape = value
        self.width = value[1]
        self.height = value[0]
        self.define_k_matrix()

    def define_k_matrix(self):
        """
        Defines the K & Kinv matrices (affine transformations) to project from pixel (u, v)
        to (theta, phi)
        """
        # ! Kinv (u,v) --> (theta, phi)
        # ! K (theta, phi) --> (u, v)
        self.Kinv = np.asarray(
            (2 * np.pi / self.width, 0, -np.pi,
             0, -np.pi / self.height, np.pi / 2,
             0, 0, 1)).reshape(3, 3)
        self.K = np.linalg.inv(self.Kinv)

    @shape.deleter
    def shape(self):
        self.__shape = (512, 1024)
        self.width = self.__shape[1]
        self.height = self.__shape[0]

    # endregion
    def __init__(self, shape=(512, 1024), cam=None):
        super(Equirectangular, self).__init__()
        if cam is None:
            self.shape = shape

        elif cam.__class__.__name__ == "Equirectangular":
            self.shape = cam.shape

        else:
            raise ValueError('Invalid Equirectangular camera model')

    def pixel2euclidean_space(self, pixels):
        """
        From a set of equirectangular pixel, this returns the normalized vector representation for those pixels.
        Such normalization corresponds to the vectors which lay in sphere ratio=1
        """
        assert pixels.shape[1] in (2, 3), "pixels parameter out of shape (n, 3) or (n, 2). We got {}".format(
            pixels.shape)
        # TODO This procedure is repeated either in Pinhole as UnifiedModel, and spherical too. Maybe we could centralize it into CameraBase
        # Getting of pixels as [u, v, 1]
        # ! We add 0.5 to compensate pixels digitization (quatization)
        pixels_corrd = np.hstack([pixels[:, 0:2]+0.5, np.ones((len(pixels[:, 0]), 1))])

        sphere_coords = np.dot(pixels_corrd, self.Kinv.T)
        return self.sphere2bearing(sphere_coords.T)

    def sphere2bearing(self, sphere_coords):
        sin_coords = np.sin(sphere_coords[0:2, :])
        cos_coords = np.cos(sphere_coords[0:2, :])

        return np.vstack(
            [
                sin_coords[0, :] * cos_coords[1, :],
                - sin_coords[1, :],
                cos_coords[0, :] * cos_coords[1, :]
            ])

    def project2equirect(self, pcl, color, rot_err_deg=1, empty_fill=0):
        """
        Projects a pcl (n, 3) which matches with color array into a equirectangular image defined by self.shape
            pcl: [N, 3] should be on unit sphere
            color: [N, *]
        """
        assert len(pcl.shape) == 2 and pcl.shape[1] == 3
        assert len(color.shape) == 2

        k_value = 2  # set the k value for knn
        n_ch = color.shape[1]
        rgb = np.concatenate([color, [np.zeros(n_ch) + empty_fill]], axis=0)
        diff = 2 * np.sin(rot_err_deg * np.pi / 180)
        # diff = 0.05
        # for origin length xyz
        old_xyz = np.concatenate([pcl, [np.zeros(n_ch) + empty_fill]], axis=0)
        # force on unit sphere
        xyz = pcl / np.sqrt((pcl ** 2).sum(1, keepdims=True))
        equirec_xyz = self.uv2unitxyz(self.equirect_uvgrid()).reshape(-1, 3)

        # print("Equirect:")
        # tic = time.time()
        tree = cKDTree(xyz, balanced_tree=False)
        dist, idx = tree.query(equirec_xyz, k=k_value, distance_upper_bound=diff, n_jobs=-1)
        # print(time.time() - tic)

        # This is the origin distance and depth for masking

        origin_dist = np.sqrt((old_xyz[idx] ** 2).sum(2, keepdims=False))
        # print('origin dist', origin_dist.shape)
        depth = np.sqrt((old_xyz ** 2).sum(1, keepdims=True))
        # print('depth', depth.shape, 'idx', idx.shape, depth[idx].shape, 'rgb', rgb.shape, rgb[idx].shape)
        # mask = origin_dist < origin_dist.sum(1, keepdims=True) / k_value
        mask_big = origin_dist > origin_dist.sum(1, keepdims=True) / k_value
        dist = dist.sum(1, keepdims=True) - dist
        dist[mask_big] = 0  # mask out the value of the neighbor points that are larger than the mean of neighbor values
        p = (dist / dist.sum(1, keepdims=True))[..., None]
        equirec = (rgb[idx] * p).sum(1).reshape(self.height, self.width, n_ch)
        equirec = np.clip(equirec, rgb.min(), rgb.max())

        return equirec.astype(np.uint8)

    def project2sparsedepthmap(self, pcl, color, rot_err_deg=1, empty_fill=0):

        xyz = pcl
        rgb = color
        pos = xyz / np.sqrt((xyz ** 2).sum(1, keepdims=True))
        equirec_xyz = self.uv2unitxyz(self.equirect_uvgrid()).reshape(-1, 3)
        # print pos.shape,"pos shape"

        # tree for knn
        k_value = 30  # the value for knn's k
        rot_err_deg = 1
        empty_fill = 0
        diff = 2 * np.sin(rot_err_deg * np.pi / 180)
        tree = cKDTree(pos, balanced_tree=False)
        dist, idx = tree.query(pos, k=k_value, distance_upper_bound=diff, n_jobs=-1)
        # print('xyz shape', xyz.shape, 'dist shape', dist.shape, 'idx shape', idx.shape)
        xyz = np.concatenate([xyz, [np.zeros(3) + empty_fill]], axis=0)
        # print(xyz[idx].shape)
        depth_nn = np.sqrt((xyz[idx] ** 2).sum(2, keepdims=False))
        depth_avg = depth_nn.sum(1, keepdims=True) / k_value
        depth_max = np.amax(depth_nn, 1, keepdims=True)
        # print('depth max shape', depth_max.shape)

        # uv = np.zeros((2,pc.shape[1]))
        # print uv.shape,"uvshape"
        uv = np.zeros((pcl.shape[0], 2))

        # maskzpnp = (pos[2, :] >= 0) & (pos[0, :] >= 0)
        # maskzp = (pos[2, :] >= 0)
        # maskznxp = (pos[2, :] < 0) & (pos[0, :] >= 0)
        # maskznxn = (pos[2, :] < 0) & (pos[0, :] < 0)
        maskzpnp = (pos[:, 2] >= 0) & (pos[:, 0] >= 0)
        maskzp = (pos[:, 2] >= 0)
        maskznxp = (pos[:, 2] < 0) & (pos[:, 0] >= 0)
        maskznxn = (pos[:, 2] < 0) & (pos[:, 0] < 0)

        # mask
        """
        uv[0,:][maskzp] = (np.arcsin(pos[0,:]/((pos[0,:]**2 + pos[2,:]**2)**0.5)) + (math.pi))[maskzp]
        uv[0,:][maskznxp] = (-np.arcsin(pos[0,:]/((pos[0,:]**2 + pos[2,:]**2)**0.5)) + (math.pi*2 ))[maskznxp]
        uv[0,:][maskznxn] = (-np.arcsin(pos[0,:]/((pos[0,:]**2 + pos[2,:]**2)**0.5)) )[maskznxn]

        uv[1,:] = np.arccos(-pos[1,:])
        """
        uv[:, 0][maskzp] = (np.arcsin(pos[:, 0] / ((pos[:, 0] ** 2 + pos[:, 2] ** 2) ** 0.5)) + (math.pi))[maskzp]
        uv[:, 0][maskznxp] = (-np.arcsin(pos[:, 0] / ((pos[:, 0] ** 2 + pos[:, 2] ** 2) ** 0.5)) + (math.pi * 2))[
            maskznxp]
        uv[:, 0][maskznxn] = (-np.arcsin(pos[:, 0] / ((pos[:, 0] ** 2 + pos[:, 2] ** 2) ** 0.5)))[maskznxn]
        uv[:, 1] = np.arccos(-pos[:, 1])
        # np.save('uv.npy',uv)

        # uvrgb = np.concatenate((uv,rgb),0)
        # np.save('uvrgb.npy',uvrgb)

        # coord = uv / math.pi * 512 - 0.5
        coord = (uv / math.pi * 512 - 0.5).astype(int)
        # equi = np.zeros((512, 1024, 3))
        equi_de = np.zeros((512, 1024))
        # equi_de_floor = np.zeros((512,1024))
        for i in range(coord.shape[0]):
            # if coord[1,index] > -512 and coord[0,index] > -512:
            #    equi[int(coord[1,index]), int(coord[0,index]), :] = rgb[:,index]
            # equi[int(coord[1,index]), int(coord[0,index]), :] = rgb[index,:]   # for sparse rgb
            # if np.sqrt(sum(xyz[index,:]**2)) <= depth_avg[index,0]:
            # if np.sqrt(sum(xyz[index,:]**2)) < depth_max[index,0]:
            if depth_max[i, 0] - depth_avg[i, 0] > 0.05:
                if np.sqrt(sum(xyz[i, :] ** 2)) <= depth_avg[i, 0]:
                    equi_de[int(coord[i, 1]), int(coord[i, 0])] = np.sqrt(sum(xyz[i, :] ** 2))
            else:
                equi_de[int(coord[i, 1]), int(coord[i, 0])] = np.sqrt(sum(xyz[i, :] ** 2))

            # equi_de_floor[int(coord[1,index]),int(coord[0,index])] = np.sqrt(sum(xyz[index,:]**2)) # for non-fixed sparse equi depth
        # imsave('equi.png',equi)
        # imsave('equi_de.png',equi_de)
        # imsave('equi_de_floor.png',equi_de_floor)
        return equi_de

    def equirect_uvgrid(self):
        u = np.linspace(-np.pi, np.pi, num=self.width, dtype=np.float32)
        v = np.linspace(-np.pi, np.pi, num=self.height, dtype=np.float32) / 2

        return np.stack(np.meshgrid(u, v), axis=-1)

    def uv2unitxyz(self, uv):
        u, v = np.split(uv, 2, axis=-1)
        y = np.sin(v)
        c = np.cos(v)
        x = c * np.sin(u)
        z = c * np.cos(u)

        return np.concatenate([x, y, z], axis=-1)

    def depthmap2pcl(self, depthmap, scaler=1, return_hm=False, return_msk=False):

        z = depthmap.reshape((-1, 1))
        mask = z > 0
        pts_xyz_on_sphere = self.uv2unitxyz(self.equirect_uvgrid()).reshape(-1, 3)

        pts_3D = np.multiply(pts_xyz_on_sphere, z) * scaler
        pts_3D = pts_3D[mask.reshape((-1,)), :]

        if return_hm:
            pts_3D = np.vstack((pts_3D, np.ones_like(pts_3D)))
        if return_msk:
            return pts_3D, mask.reshape((-1,))
        return pts_3D

    def depthmap2colorpcl(self, depthmap, colormap, scaler=1, format='rgb'):

        pts_3D, mask = self.depthmap2pcl(depthmap, scaler=scaler, return_msk=True)

        if len(colormap.shape) == 3 and format is 'rgb':
            color = np.zeros((3, len(pts_3D[:, 0])))
            color[0, :] = colormap[:, :, 2].reshape((1, -1))[0, mask]
            color[1, :] = colormap[:, :, 1].reshape((1, -1))[0, mask]
            color[2, :] = colormap[:, :, 0].reshape((1, -1))[0, mask]
        elif len(colormap.shape) == 3 and format is 'gbr':
            color = np.zeros((3, len(pts_3D[:, 0])))
            color[0, :] = colormap[:, :, 1].reshape((1, -1))[0, mask]
            color[1, :] = colormap[:, :, 0].reshape((1, -1))[0, mask]
            color[2, :] = colormap[:, :, 2].reshape((1, -1))[0, mask]
        else:
            color = colormap[:, :].reshape((1, -1))[0, mask]

        return pts_3D, color.T

    def spherexyz2equirec(self, xyz, color, rot_err_deg=0.5, empty_fill=0):
        """

        """
        assert len(xyz.shape) == 2 and xyz.shape[1] == 3
        assert len(color.shape) == 2

        h, w = self.get_shape()
        n_ch = color.shape[1]
        rgb = np.concatenate([color, [np.zeros(n_ch) + empty_fill]], axis=0)
        diff = 2 * np.sin(rot_err_deg * np.pi / 180)

        # force on unit sphere
        xyz = xyz / np.sqrt((xyz ** 2).sum(1, keepdims=True))
        equirec_xyz = self.uv2unitxyz(self.equirect_uvgrid()).reshape(-1, 3)

        tree = cKDTree(xyz, balanced_tree=True)
        dist, idx = tree.query(equirec_xyz, k=5, distance_upper_bound=diff, n_jobs=-1)
        dist = dist.sum(1, keepdims=True) - dist
        p = (dist / dist.sum(1, keepdims=True))[..., None]
        equirec = (rgb[idx] * p).sum(1).reshape(h, w, n_ch)
        equirec = np.clip(equirec, rgb.min(), rgb.max())

        return np.nan_to_num(equirec)
