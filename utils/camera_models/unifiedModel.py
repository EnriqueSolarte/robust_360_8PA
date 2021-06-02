from abc import ABC

import numpy as np

from utils.camera_models import Equirectangular
from utils.camera_models import Pinhole
from utils.camera_models import Camera
import cv2
from utils.geometry_utilities import *


class UnifiedModel(Camera):
    def define_k_matrix(self):
        self.K = self.camera_pinhole.K
        self.Kinv = self.Kinv

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
        self.hight = value[0]
        self.camera_pinhole.shape = value

    # endregion

    @classmethod
    def by_calibrated_parameters(cls, calib, cam_label="cam0"):
        intrinsics = calib[cam_label]["intrinsics"]
        distortion_coeffs = calib[cam_label]["distortion_coeffs"]
        resolution = calib[cam_label]["resolution"]
        distortion_model = calib[cam_label]["distortion_model"]
        camera_model = calib[cam_label]["camera_model"]
        cam = None
        if camera_model == "omni":
            param = dict(fx=intrinsics[1],
                         fy=intrinsics[2],
                         cx=intrinsics[3],
                         cy=intrinsics[4],
                         eps=intrinsics[0],
                         width=resolution[1],
                         hight=resolution[0])
            cam = cls(**param)
            cam.distortion_coeffs = np.array(distortion_coeffs)
        cam.calib = calib
        return cam

    def __init__(self, fx=None, fy=None, cx=None, cy=None, eps=None, width=None, hight=None, cam=None):
        super(UnifiedModel, self).__init__()
        if cam.__class__.__name__ == "UnifiedModel":
            self.eps = cam.eps
            self.camera_pinhole = cam.pinhole_model
            self.shape = cam.shape

        else:
            assert None not in [fx, fy, cx, cy, eps, width, hight]
            self.eps = eps
            self.camera_pinhole = Pinhole(**dict(fx=fx,
                                                 fy=fy,
                                                 cx=cx,
                                                 cy=cy,
                                                 width=width,
                                                 hight=hight))
            self.shape = (hight, width)
            self.define_k_matrix()

    @staticmethod
    def pcl2sphere(pcl):
        """
        Back-projects a pcl array (N, 4) or (N, 3) onto a spherical surface |r| = 1
        """
        # TODO this is a fast implementation. It needs to be polished

        assert pcl.shape[1] in [4, 3], "Wrong pcl.shape, expected (n, 4) or (n, 3), got it {}".format(pcl.shape)

        rNorm = np.linalg.norm(pcl[:, 0:3], axis=1)
        # assert 0 not in rNorm
        point_3d = pcl[:, 0:3] / rNorm[:, None]

        return point_3d

    def sphere2pxl_coords(self, points_3d):
        """
        Back-projects 3d points onto a spherical surface to a image plane using K intrinsic matrix
        defined for the Unified model
        """
        assert points_3d.shape[1] in [3, 4], "Wrong points_3d.shape, expected (n, 4) or (n, 3), got it {}".format(
            points_3d.shape)

        # TODO this is a fast implementation. It needs to be polished
        hm_points = self.homogeneous_normalization(points_3d)
        px_points = np.round(np.dot(hm_points, self.camera_pinhole.K.T)).astype(np.int)

        mask_zero = px_points >= 0
        mask_width = px_points[:, 0] < self.width - 1
        mask_hight = px_points[:, 1] < self.hight - 1
        valid_mask = mask_zero[:, 0] * mask_zero[:, 1] * mask_hight * mask_width

        return px_points, valid_mask

    def pcl2pcl_on_sphere(self, pcl):
        """
        Based on a set of points in 3d, it returns its projection on a sphere of ratio 1
        """
        assert_plc_shape(pcl)
        pcl[:, 2] -= self.eps_vector(len(pcl[:, 0]))

        mask = pcl[:, 2] > 0
        pcl = pcl[mask]
        norm_pcl = np.linalg.norm(pcl[:, 0:3], axis=1)

        return pcl[:, 0:3] / norm_pcl[:, None]

    def pixel2euclidean_space(self, px_points, undistorted=True):
        """
        Projects a set of pixels (N, 2) or (N, 3) to a shifted spherical surface (see. unified model)
        """
        assert px_points.shape[1] in [2, 3, 4]

        if undistorted:
            px_points_un = self.undistort_points(px_points)
        else:
            px_points_un = px_points.copy()

        if px_points_un.shape[1] == 2:
            px_points_un = extend_array_to_homogeneous(px_points_un)

        hm_points = np.dot(px_points_un, self.camera_pinhole.Kinv.T)
        eps_array = np.ones((len(hm_points[:, 0]), 1)) * self.eps
        u2_v2 = (hm_points[:, 0] ** 2 + hm_points[:, 1] ** 2)[:, None]
        eps = 1 - eps_array ** 2
        sqrt__ = np.sqrt((1 + eps * u2_v2))

        unified_factor = (eps_array + sqrt__) / (u2_v2 + 1)
        pts_on_sphere = unified_factor * hm_points
        pts_on_sphere[:, 2] -= eps_array[-1,]

        return pts_on_sphere.T

    def pcl2pxl_coords(self, pcl):
        """
        Projects a pcl (3d or 4d-homogeneous points) to a pixels coordinates. The
        PCL must be referenced at camera coordinates. The projection follows the projection described
        for the unified model; first to a sphere shifted eps times from the camera coordinates, then
        a pinhole projection onto a image plane.\n
        This implementation is based on the equation (4) defined in
        [1] D. Caruso, J. Engel, and D. Cremers, “Large-scale direct SLAM for omnidirectional cameras,
        ” IEEE Int. Conf. Intell. Robot. Syst., vol. 2015-Decem, pp. 141–148, 2015.
        """

        normPCL = np.linalg.norm(pcl[:, 0:3], axis=1)
        denominator = pcl[:, 2] + normPCL * self.eps_vector(len(pcl[:, 0]))
        hm_points = pcl[:, 0:2] / denominator[:, None]
        hm_points = np.hstack([hm_points, np.ones_like(hm_points)])[:, 0:3]
        pxl_points = np.round(np.dot(hm_points, self.camera_pinhole.K.T)).astype(np.int)

        idx = np.dot(pxl_points, self.camera_pinhole.W.T).astype(np.int)

        return pxl_points, idx

    def pcl2image_plane(self, pcl):
        """
           Given a PCL (point cloud [N, 3] or [N, 4]) both, a mask and an image are evaluated.
           The image represents the image plane projection defined defined by the Unified Model.
           The mask is the boolean array for the points PCL which are truly projected into a image plane.
        """
        mask_valid_pts = self.masking_pts(pcl)
        # pcl = pcl[mask_valid_pts, :]
        px_pts, idx = self.pcl2pxl_coords(pcl=pcl)
        mask = self.camera_pinhole.masking_coords(px_pts) * mask_valid_pts
        image = np.zeros((self.get_shape())).reshape(-1)
        idx = idx[mask]
        image[idx] = 1
        return image.reshape(self.get_shape()), mask

    def masking_pts(self, pcl):
        """
        This function computes a mask for all pts in front a camera location
        regarding a Unified model.
        """
        assert pcl.shape[1] in [3, 4], "PLC shape does not match. Expected (n, 3) or (n, 4). We got {}".format(
            pcl.shape)

        pcl = np.copy(pcl)
        pcl[:, 2] -= self.eps_vector(len(pcl[:, 0]))
        return pcl[:, 2] > 0

    def eps_vector(self, length):
        return np.ones((length,)) * self.eps

    # TODO maybe this function is already defined in PinholeModel
    # def pixels2homogeneous(self, pixels_coord):
    #     """
    #     Projects a set of pixels coord in homogeneous representation
    #     """
    #     assert pixels_coord.shape[1] in [2, 3]
    #
    #     if pixels_coord.shape[1] == 2:
    #         pixels_coord = np.hstack([pixels_coord, np.ones((len(pixels_coord[:, 0]), 1))])
    #     hm_points = np.dot(pixels_coord, self.Kinv.T)
    #
    #     return hm_points

    def unified_model2equirectangular(self, image, mask, shape):
        """
        Returns a set of vectores 3R onto a hemisphere (unified model) given a image array (h, w) (fish-eye image)
        :return:
        """
        assert image.shape[0] == self.get_shape()[0] and image.shape[1] == self.get_shape()[1], "{}!={}".format(
            image.shape, self.get_shape())

        pts_sphere, color = self.get_shifted_sphere_from_fisheye_image(
            image=image,
            mask=mask
        )

        camera_equirect = Equirectangular(shape=shape)
        equirect_img = camera_equirect.spherexyz2equirec(pts_sphere, color, rot_err_deg=1)

        return (equirect_img * 255).astype(np.uint8)

    def get_shifted_sphere_from_fisheye_image(self, image, mask=None, plot=False):

        if mask is None:
            try:
                mask = np.ones_like(image[:, :, 0])
            except:
                mask = np.ones_like(image)

        pts_px, idx = self.camera_pinhole.get_pxl_coords_by_image_mask(mask)

        pts_sphere = self.pxl2shifted_sphere(pts_px)
        color = self.get_color_array(image).astype(np.float) / 255
        if plot:
            plot_color_plc(pts_sphere, color)
        return pts_sphere, color

    def undistort_points(self, pxl):
        pxl_ = pxl.reshape((-1, 1, 2)).astype(np.float32)
        u_pxl = cv2.undistortPoints(
            src=pxl_,
            cameraMatrix=self.K,
            distCoeffs=self.distortion_coeffs,
            R=None,
            P=self.K
        ).reshape((-1, 2))
        return u_pxl
