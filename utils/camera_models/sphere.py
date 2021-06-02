import numpy as np
from utils.vispy_utilities import *
from scipy.spatial import cKDTree
import time
from utils.camera_models import Pinhole
from utils.camera_models import Equirectangular
from utils.camera_models import Camera


class Sphere(Camera):

    def define_k_matrix(self):
        self.K = self.camera_equirectangular.K
        self.Kinv = self.camera_equirectangular.Kinv

    # region shape property
    @property
    def shape(self):
        return self.__shape

    @shape.setter
    def shape(self, value):
        assert isinstance(value, tuple)
        assert len(value) == 2
        # print(value)
        self.__shape = value
        self.width = value[1]
        self.height = value[0]
        self.camera_equirectangular.shape = value
        self.grid = self.equirectangular_grid(value)
        self.grid2 = np.squeeze(np.stack([self.grid[0].reshape(-1, 1), self.grid[1].reshape(-1, 1)], axis=1))
        self.vgrid = self.vector_grid(self.grid)
        self.px_grid = self.pixel_grid()
        self.tree = cKDTree(self.grid2, balanced_tree=False)
        self.define_default_bearings(value)

    @shape.deleter
    def shape(self):
        self.__shape = (512, 1024)
        self.width = self.__shape[1]
        self.height = self.__shape[0]

    # endregion

    def __init__(self, cam=None, shape=(512, 1024)):
        super(Sphere, self).__init__()
        # TODO add unified model to sphere camera model
        # ! Maybe a pinhole camera inside of Sphere() does not make sense
        # ! neither into Equirectangular()
        if cam is None:
            # print("Sphere camera defined")
            self.camera_pinhole = None
            self.camera_equirectangular = Equirectangular(shape=shape)
            self.sphere_ratio = 1
            # ! shape is a property defined.
            # ! it sets a grid, internal shape, and KdTree based on
            # ! the given shape
            self.shape = shape
        elif cam.__class__.__name__ == "Sphere":
            print("Defining cam from previous Sphere camera model")
            self.camera_pinhole = cam.camera_pinhole
            self.sphere_ratio = cam.shere_ratio
            self.shape = cam.shape
        else:
            print("The given obj-camera parameter to construct sphere obj-camera", cam)
            raise AttributeError("Invalid camera obj to construct Sphere camera model")

    def define_default_bearings(self, shape):
        h, w = shape
        u = np.linspace(0, w - 1, w).astype(int)
        v = np.linspace(0, h - 1, h).astype(int)
        uu, vv = np.meshgrid(u, v)
        self.deaful_pixel = np.vstack((uu.flatten(), vv.flatten(), np.ones((w * h,)))).astype(np.int)
        self.default_spherical = self.camera_equirectangular.Kinv.dot(self.deaful_pixel)
        self.default_bearings = self.camera_equirectangular.sphere2bearing(self.default_spherical)

    def pixel2euclidean_space(self, pixels):
        """
        It is not clear yet how a set of pixel can be described into a sphere.
        Therefore, here the Equirectangular camera is used
        for calculating such norm vectors, assuming that the given pixels
        are coming from a equirectangular images.
        """
        return self.camera_equirectangular.pixel2euclidean_space(pixels)

    @classmethod
    def from_pinhole(cls, cam):
        """
        Alternative constructor for Sphere class using Pinhole class
        """
        assert cam.__class__.__name__ == "Pinhole"
        _camera_sphere = cls()
        _camera_sphere.camera_pinhole = cam
        return _camera_sphere

    def fast_pinhole_camera(self, image_shape):
        cam = Pinhole(fx=image_shape[1] / 2,
                      fy=image_shape[0] / 2,
                      cx=image_shape[1] / 2,
                      cy=image_shape[0] / 2,
                      width=image_shape[1],
                      hight=image_shape[0])

        self.camera_pinhole = cam

    def get_shape(self):
        return self.shape

    def image2sphere(self, image):
        """

        :param image:
        :return:
        """
        rgb_image = len(image.shape) > 2
        if rgb_image:
            h, w, c = image.shape
            hm_coord = self.camera_pinhole.constructNormalizedCoordinate(image[:, :, 1])
        else:
            h, w = image.shape
            c = 1
            hm_coord = self.camera_pinhole.constructNormalizedCoordinate(image)

        hm_coord = np.dstack([hm_coord, np.ones((h, w))])
        hm_coord_norm = np.linalg.norm(hm_coord, axis=2)
        hm_coord_unit_vectors = np.zeros((h, w, c + 3))
        hm_coord_unit_vectors[:, :, 0] = np.divide(hm_coord[:, :, 0], hm_coord_norm) * self.sphere_ratio
        hm_coord_unit_vectors[:, :, 1] = np.divide(hm_coord[:, :, 1], hm_coord_norm) * self.sphere_ratio
        hm_coord_unit_vectors[:, :, 2] = np.divide(hm_coord[:, :, 2], hm_coord_norm) * self.sphere_ratio
        if rgb_image:
            hm_coord_unit_vectors[:, :, 3:] = image
        else:
            hm_coord_unit_vectors[:, :, 3] = image

        return hm_coord_unit_vectors

    def plot_image_sphere(self, sphere, return_val=False):
        assert len(sphere.shape) == 3
        pcl = sphere[:, :, 0:3].reshape((-1, 3))
        if sphere.shape[2] > 4:
            color = sphere[:, :, 3:].reshape((-1, 3))
        else:
            color = sphere[:, :, 3].reshape((-1, 1))
            color = np.hstack([color, color, color])

        if return_val:
            return pcl, color

        plot_color_plc(points=pcl, color=color)

    def pinhole2homogeneous_coords(self, image, return_gray_scale=False):

        assert self.camera_pinhole is not None, \
            "Pinhole camera has not been instanced \n" \
            "You want to run def_fast_pinhole_camera for a fast camera definition"

        if len(image.shape) > 2:
            mask = image[:, :, 0] > -1
        else:
            mask = image[:, :] > -1

        pts_px = np.array(np.where(mask))
        pts_px = np.vstack((pts_px[1, :], pts_px[0, :], np.ones((1, len(pts_px[1])))))

        pts_hm = np.matmul(self.camera_pinhole.Kinv, pts_px)
        hm_coord_norm = np.linalg.norm(pts_hm, axis=0)
        hm_coord_unit_vectors = np.divide(pts_hm, hm_coord_norm) * self.sphere_ratio
        if len(image.shape) > 2:
            color = image.reshape((-1, 3))
            return hm_coord_unit_vectors.T, color
        else:
            color = image.reshape((-1, 1))
            if return_gray_scale:
                return hm_coord_unit_vectors.T, color
            else:
                return hm_coord_unit_vectors.T, np.hstack([color, color, color])

    def equirectangular2sphere(self, image):
        """
        Projects an equirectangular image (h, w, c) onto a unit surface sphere
        :param image: Equirectangular image. image.shape == self.get_shape()
        :return: pcl, color: both in (n, 3)
        """
        if len(image.shape) > 2:
            h, w, c = image.shape
        else:
            h, w = image.shape

        assert (h, w) == self.get_shape()

        tilde_theta, tilde_phi = self.equirectangular_grid((h, w))

        y = -np.sin(tilde_phi) * self.sphere_ratio
        x = np.sin(tilde_theta) * np.cos(tilde_phi) * self.sphere_ratio
        z = np.cos(tilde_theta) * np.cos(tilde_phi) * self.sphere_ratio

        pcl = np.hstack([x.reshape(-1, 1), y.reshape(-1, 1), z.reshape(-1, 1)])

        if len(image.shape) > 2:
            color = image.reshape((-1, 3)).astype(np.float) / 255.0
        else:
            color = image.reshape((-1, 1)).astype(np.float) / 255.0
            color = np.concatenate([color, color, color], axis=1)

        return pcl, color

    def equirectangular2unified_model(self, image, unified_model, format=np.float, rearView=False):
        """
        Maps an equirectangular image described by a sphere projection into a fish-eye image by The Unified Model
        :param image: equirectangular image, image.shape == self.get_shape()
        :return: image: fish-eye image, image.shape == unified_model.get_shape()
        """
        pcl, color = self.equirectangular2sphere(image)

        # masking hemisphere
        if rearView:
            mask = pcl[:, 2] < 0
        else:
            mask = pcl[:, 2] > 0

        pcl = pcl[mask]
        color = color[mask]
        RBG_color = np.zeros_like(color)
        RBG_color[:, 0] = color[:, 2]
        RBG_color[:, 1] = color[:, 1]
        RBG_color[:, 2] = color[:, 0]
        # Adding eps from Unified model definition
        if rearView:
            pcl[:, 2] *= -1

        pcl[:, 2] += unified_model.eps_vector(len(pcl[:, 2]))
        # plot_color_plc(pcl, RBG_color)

        if format is np.float:
            if rearView:
                return np.fliplr(unified_model.camera_pinhole.colorpcl2rgbmap(pcl.T, color.T, format=format))
            return unified_model.camera_pinhole.colorpcl2rgbmap(pcl.T, color.T, format=format)
        else:
            if rearView:
                return np.fliplr(unified_model.camera_pinhole.colorpcl2rgbmap(pcl.T, color.T, format=format) * 255)
            return unified_model.camera_pinhole.colorpcl2rgbmap(pcl.T, color.T, format=format) * 255

    def pinhole2equirectangular(self, image, shape=(512, 1024), pose=None, gray_scale=True):
        hm_pts, color = self.pinhole2homogeneous_coords(image, return_gray_scale=gray_scale)

        if pose is not None:
            hm_pts = np.dot(np.hstack([hm_pts, np.ones((len(hm_pts[:, 0]), 1))]), pose.T)[:, 0:3]

        from projections.spherexyz2equirec import spherexyz2equirec
        equirect_img = spherexyz2equirec(hm_pts, color, rot_err_deg=0.1, w=shape[1], h=shape[0])

        return np.squeeze(equirect_img)

    def pixel_grid(self, shape=None):

        if shape is None:
            shape = self.shape

        phi = np.linspace(0, shape[0] - 1, shape[0]).astype(np.int)
        theta = np.linspace(0, shape[1] - 1, shape[1]).astype(np.int)
        tilde_theta, tilde_phi = np.meshgrid(theta, phi)

        return np.hstack([tilde_theta.reshape(-1, 1), tilde_phi.reshape(-1, 1)]).astype(np.int)

    def vector_grid(self, grid):
        tilde_theta = grid[0]
        tilde_phi = grid[1]

        y = -np.sin(tilde_phi) * self.sphere_ratio
        x = np.sin(tilde_theta) * np.cos(tilde_phi) * self.sphere_ratio
        z = np.cos(tilde_theta) * np.cos(tilde_phi) * self.sphere_ratio

        return np.hstack([x.reshape(-1, 1), y.reshape(-1, 1), z.reshape(-1, 1)])

    def vector_grid2equirect_grid(self, vector_grid):

        # Check if vector_grid elements are unit vectors
        assert np.isclose(np.linalg.norm(vector_grid[np.random.randint(0, len(vector_grid[:, 1])), :]), 1)

        theta = np.arctan2(vector_grid[:, 0], vector_grid[:, 2])
        phi = np.arcsin(-vector_grid[:, 1])

        return np.stack([theta, phi], axis=1)

    def equirectangular_grid(self, shape=None):
        if shape is None:
            shape = self.shape

        phi_step = np.pi / shape[0]
        theta_step = 2 * np.pi / shape[1]
        phi = np.linspace(np.pi / 2, -np.pi / 2 + phi_step, shape[0]-1)
        theta = np.linspace(-np.pi, np.pi - theta_step, shape[1]-1)
        tilde_theta, tilde_phi = np.meshgrid(theta, phi)

        return tilde_theta, tilde_phi

    def image_pinhole2sphere(self, image):
        shape = image.shape
        hm_pts, color = self.pinhole2homogeneous_coords(image)

        theta_coord = np.arctan(hm_pts[:, 0] / hm_pts[:, 2])
        phi_coord = np.arcsin(hm_pts[:, 1])

        import matplotlib.pyplot as plt
        plt.plot(phi_coord)

        if shape.__len__() > 2:
            img = np.vstack([theta_coord, phi_coord, color])
        else:
            img = np.vstack([theta_coord, phi_coord, color[:, 0]])

        return img

    def projectPointCloud(self, pcl):
        """Projects a pcl [n, 3] into a equirectangular image plane. 
        Returns: coords --> (u, v), idx
        """

        # PCL to homogeneous coord normalized to a unit sphere
        norm = np.linalg.norm(pcl[:, 0:3], axis=1).reshape((-1, 1))
        pts_vectors = np.divide(pcl[:, 0:3], norm) * self.sphere_ratio

        # pts_sphere: array of THETA, PHI
        pts_sphere = self.vector_grid2equirect_grid(pts_vectors)

        diff = 5 * np.sin(1 * np.pi / 180)
        # print("sphere:")
        # tic = time.time()
        dist, idx = self.tree.query(pts_sphere, k=1, distance_upper_bound=diff, n_jobs=-1)
        # print(time.time() - tic)
        #
        # print("ckDTree2:")
        # tic = time.time()
        # dist, idx = self.tree2.query(pts_vectors, k=1, distance_upper_bound=diff, n_jobs=-1)
        # print(time.time() - tic)

        coords = self.px_grid[idx]
        return coords, idx

    def warpimage(self, pcl, pose_rel, img_curr, return_mask=False, back_projection=False):

        assert (img_curr.shape[0], img_curr.shape[1]) == self.get_shape()

        pcl_warped = np.dot(pcl, pose_rel.T)

        img_corrds, idx = self.projectPointCloud(pcl)
        img_corrds_warped, idx_warp = self.projectPointCloud(pcl_warped)

        coords = [img_corrds, img_corrds_warped]

        img_warped = np.zeros_like(img_curr)
        if back_projection:
            img_warped[coords[0][:, 1], coords[0][:, 0]] = img_curr[coords[1][:, 1], coords[1][:, 0]]
        else:
            img_warped[coords[1][:, 1], coords[1][:, 0]] = img_curr[coords[0][:, 1], coords[0][:, 0]]

        if return_mask:
            return img_warped, coords
        else:
            return img_warped

    def plc2image_plane(self, pcl):
        """
        Given a PCL (n, 3) or (n, 4) a equirectangular image with projected points is defined
        :param pcl: Point Cloud (n, 3) or (n, 4)
        :return: equirectangular image: (h, w)
        """
        px_pts, idx = self.projectPointCloud(pcl=pcl)
        image = np.zeros((self.get_shape())).reshape(-1)
        image[idx] = 1
        return image.reshape(self.get_shape())

    def unified_model_mask(self, unified_model):
        """
        Returns a mask (h, w) defined by a given unified_model camera object. This mask truly represents
        all of the pixels which can be projected into an image plane
        """
        image = np.ones(self.get_shape()) * 255
        mask = self.equirectangular2unified_model(image, unified_model, format=np.uint8)
        return mask[:, :, 0]
