from utils.image_utilities import *
from utils.camera_models import Camera
from scipy.spatial import cKDTree
from scipy.interpolate import interp2d


class Pinhole(Camera):

    def define_k_matrix(self):
        self.K = np.asarray((self.fx, 0, self.cx, 0, self.fy, self.cy, 0, 0, 1)).reshape(3, 3)
        self.Kinv = np.linalg.inv(self.K)

    def pixel2euclidean_space(self, pixels):
        return self.pixels2homogeneous(pixels)

    @classmethod
    def by_camera_matrix_and_distortions(cls, k_matrix, dist, shape):
        param = dict(fx=k_matrix[0, 0],
                     fy=k_matrix[1, 1],
                     cx=k_matrix[0, 2],
                     cy=k_matrix[1, 2],
                     width=shape[1],
                     height=shape[0])
        cam = cls(**param)
        cam.distortion_coeffs = dist
        return cam

    @classmethod
    def by_K_matrix(cls, K, shape):
        assert K.shape == (3, 3)
        param = dict(fx=K[0, 0],
                     fy=K[1, 1],
                     cx=K[0, 2],
                     cy=K[1, 2],
                     width=shape[1],
                     height=shape[0],)
        cam = cls(**param)
        return cam

    @classmethod
    def by_calibrated_parameters(cls, calib):
        intrinsics = calib["cam0"]["intrinsics"]
        distortion_coeffs = calib["cam0"]["distortion_coeffs"]
        resolution = calib["cam0"]["resolution"]
        distortion_model = calib["cam0"]["distortion_model"]
        camera_model = calib["cam0"]["camera_model"]
        cam = None
        if camera_model == "omni":
            param = dict(fx=intrinsics[1],
                         fy=intrinsics[2],
                         cx=intrinsics[3],
                         cy=intrinsics[4],
                         width=resolution[1],
                         height=resolution[0])
            cam = cls(**param)
            cam.distortion_coeffs = distortion_coeffs

        return cam

    def __init__(self, fx=None, fy=None, cx=None, cy=None, width=None, hight=None, cam=None, shape=None):
        """
        Perspective camera model (pinhole model)
        :param fx: focal length x
        :param fy: focal length y
        :param cx: principal point x
        :param cy: principal point y
        :param width: image width
        :param height: image height
        :param cam: camera obj to initialize this new instance. Only Pinhole class is allowed
        """
        super(Pinhole, self).__init__()
        if cam is None:
            if None not in [fx, fy, cx, cy, width, hight]:
                self.fx = fx
                self.fy = fy
                self.cx = cx
                self.cy = cy
                self.shape = (hight, width)
            else:
                assert shape is not None
                self.cx = shape[1] // 2
                self.cy = shape[0] // 2
                self.fx = shape[1] // 2
                self.fy = shape[0] // 2
                self.shape = shape

        elif cam.__class__.__name__ == "Pinhole":
            self.fx = cam.fx
            self.fy = cam.fy
            self.cx = cam.cx
            self.cy = cam.cy
            self.shape = cam.shape

        else:
            raise ValueError('Invalid PINHOLE model')

        self.define_k_matrix()
        self.define_default_homogeneous(self.shape)

        self.W = np.array([1, self.width, 0]).reshape((1, 3))

        self.zero_vector = np.zeros((4, 1)).astype(np.float)
        self.zero_vector[3, 0] = 1

        if self.width >= self.height:
            self.image_scaling = self.width
        else:
            self.image_scaling = self.height

        self.isCameraBase = True

        self.distortion_coeffs = None
        # ! Projection marix
        self.Kinv = np.linalg.inv(self.K)

    def define_default_homogeneous(self, shape):
        h, w = shape
        u = np.linspace(0, w - 1, w).astype(int)
        v = np.linspace(0, h - 1, h).astype(int)
        uu, vv = np.meshgrid(u, v)
        self.deafult_pixel = np.vstack((uu.flatten(), vv.flatten(), np.ones((w * h,)))).astype(np.int)
        self.default_homogeneous = self.Kinv.dot(self.deafult_pixel)

    def get_focal(self):
        return np.average([self.fx, self.fy])

    def get_pxl_coords_by_image_mask(self, mask):
        """
        Returns the pixels corrdinates (u, v, 1) which are defined as True in a mask
        """
        assert mask.shape == self.get_shape(), "{}".format(mask.shape)
        pts_px = np.array(np.where(mask))
        pts_px = np.vstack((pts_px[1, :], pts_px[0, :], np.ones((len(pts_px[1]),))))
        idx = np.matmul(self.W, pts_px).astype(np.int)[0]

        return pts_px.T, idx

    def depthmap2colorpcl(self, depthmap, colormap, scaler=1, idx_matching=False, format='rgb'):
        mask = depthmap > 0

        pts_px = np.array(np.where(mask))
        pts_px = np.vstack((pts_px[1, :],
                            pts_px[0, :],
                            np.ones((1, len(pts_px[1]))))).astype(int)

        if len(colormap.shape) == 3 and format is 'rgb':
            color = np.zeros((3, len(pts_px[0, :])))
            color[0, :] = colormap[:, :, 2][pts_px[1, :], pts_px[0, :]]
            color[1, :] = colormap[:, :, 1][pts_px[1, :], pts_px[0, :]]
            color[2, :] = colormap[:, :, 0][pts_px[1, :], pts_px[0, :]]
        else:
            color = colormap[pts_px[0, :], pts_px[1, :]]

        pts_hm = np.matmul(self.Kinv, pts_px)

        pts_3D = ((1 / scaler) * depthmap[mask]) * pts_hm
        pts_3D = np.vstack((pts_3D, np.ones((1, len(pts_3D[1])))))

        if idx_matching:
            idx = np.where(np.reshape(mask, (mask.size)))[0].astype(np.int)
            return pts_3D, color, idx
        return pts_3D, color / 255

    def get_pixels_pos_from_pcl(self, pcl):
        z = pcl[2, :]
        pts_hm = pcl[0:3, :] / z
        pts_px = (np.round(np.matmul(self.K, pts_hm)))

        return pts_px

    def image_alignment(self, pcl, image, idx, return_idxs=False, return_pixels_pos=False, norm=False, type=np.uint8):
        """
        This method allows us to project a point cloud PCL into an image given its indexing points to a certain image.
        The idx tells us which 3D point is related to which pixel in the image.
        Is important to note a certain pcl could not be totally projected into a image of (height, width). For that new
        indexes are calculated. In deed new_idxs are the indexes in idx but are able to be projected into a image of
        (height, width)

        :param pcl: point cloud (4xN)
        :param idx: (1xN)
        :param image: (height, width)
        :param return_idxs: Boolean to return the new indexes
        :return: it could be just the projected image either the prejected image and the new indexes
                image: (height, width)
                new_idxs: (2,N')
        """

        if norm:
            type = np.float

        z = pcl[2, :]
        pts_hm = pcl[0:3, :] / z
        pts_px = (np.round(np.matmul(self.K, pts_hm)))

        new_idx = np.matmul(self.W, pts_px).astype(np.int)[0]

        mask_pcl = closest_pcl(pcl=pcl, idx=new_idx)

        mask_zero = pts_px >= 0
        mask_width = pts_px[0] < self.width - 1
        mask_height = pts_px[1] < self.height - 1
        mask = mask_zero[0] * mask_zero[1] * mask_height * mask_width * mask_pcl

        idx = idx[mask]
        new_idx = new_idx[mask]
        idxs = np.vstack((idx, new_idx))

        if len(image.shape) > 2:
            new_image = np.zeros((self.height, self.width, image.shape[-1]))
            for ch in range(image.shape[-1]):
                aux_image = np.zeros((1, self.height * self.width))
                aux_image[0, idxs[1]] = image[:, :, ch].reshape((1, -1))[0, idxs[0]]
                aux_image = aux_image.reshape((self.height, self.width)).astype(type)

                new_image[:, :, ch] = aux_image

        else:
            new_image = np.zeros((1, self.height * self.width))
            new_image[0, idxs[1]] = image.reshape((1, -1))[0, idxs[0]]
            new_image = new_image.reshape((self.height, self.width)).astype(type)

        if norm:
            new_image = new_image.astype(np.float) / 255

        if return_idxs and return_pixels_pos:
            return new_image.astype(type), idxs, pts_px
        elif return_pixels_pos:
            return new_image.astype(type), pts_px
        elif return_idxs:
            return new_image.astype(type), idxs
        else:
            return new_image.astype(type)

    def pcl_projection(self, pcl, idx_matching=False):
        """
        based on a PCL input (on camera frame) each point is projected on a image plane. This function output the
        pixels coordinates for the PCL input
        :param pcl: PCL [3xN] or [4xN]
        :param idx_matching: flag for returning indexes
        :return: pixels_points
                 pixels_points, indexes
        """
        mask_pts_in_front = self.masking_pts(pcl)
        z = pcl[2, :]
        pts_hm = pcl[0:3, :] / z
        pts_px = (np.round(np.dot(self.K, pts_hm)))
        if idx_matching:
            new_idx = np.dot(self.W, pts_px).astype(np.int)

            # mask_pcl = closest_pcl(pcl=pcl, idx=new_idx)
            mask_zero = pts_px >= 0
            mask_width = pts_px[0] < self.width - 1
            mask_height = pts_px[1] < self.height - 1
            mask = mask_zero[0] * mask_zero[1] * mask_height * mask_width * mask_pts_in_front

            return pts_px, new_idx.reshape((-1)), mask
        else:
            return pts_px

    def warping_back(self, pcl, idx, image_warping, idx_matching=False):
        pts_px, idx_new, mask = self.pcl_projection(pcl, idx_matching=True)

        idx = idx[mask]
        idx_new = idx_new[mask]
        idxs = np.vstack((idx, idx_new))
        new_image = np.zeros_like(image_warping)

        if len(image_warping.shape) > 2:
            for ch in range(image_warping.shape[-1]):
                aux_image = np.zeros((1, self.height * self.width))
                aux_image[0, idxs[0]] = image_warping[:, :, ch].reshape((1, -1))[0, idxs[1]]
                aux_image = aux_image.reshape((self.height, self.width))
                new_image[:, :, ch] = aux_image
        else:
            new_image = new_image.reshape((1, self.height * self.width))
            new_image[0, idxs[0]] = image_warping.reshape((1, -1))[0, idxs[1]]
            new_image = new_image.reshape((self.height, self.width))

        if idx_matching:
            return new_image, idxs, mask
        return new_image

    def pcl2depthmap(self, pcl, escaler=1, patch=(5, 3)):
        """
        This method has been updated based on
        https://github.com/ClementPinard/SfmLearner-Pytorch/blob/master/data/kitti_raw_loader.py
        https://github.com/windowsub0406/KITTI_Tutorial/blob/master/velo2cam_projection.ipynb
        """

        assert pcl.shape[0] in (3, 4)
        mask_front = pcl[2, :] > 0
        pts_px = self.projectPointCloud(pcl)

        mask = self.masking_coords(pts_px) * mask_front
        z = pcl[2, mask]
        pts_px = pts_px[:, mask]
        depthmap = np.zeros((self.height, self.width))
        depthmap[pts_px[1, :], pts_px[0, :]] = z
        #
        idx = np.where(depthmap > 0)

        def min_dist(array):
            mask = array.flatten() > 0
            return np.min(array.flatten()[mask])

        return apply_on_image(image=depthmap, patch=patch, idx=idx, function=min_dist)
        # # dupe_inds = [item for item, count in Counter(inds).items() if count > 1]
        # for dd in dupe_inds:
        #     pts = np.where(inds == dd)[0]
        #     x_loc = int(pts[0, pts[0]])
        #     y_loc = int(pts[1, pts[0]])
        #     depthmap[y_loc, x_loc] = pcl[2, pts].min()
        # depthmap[depthmap < 0] = 0

    def colorpcl2rgbmap(self, pcl, color, format=np.uint8):
        """
        Creates a image (self.height, self.width, :) given a PCL 3d and its correspond color array vector
        :param pcl:
        :param color:
        :return:
        """
        assert pcl.shape[0] in [3, 4]
        assert color.shape[0] in [1, 3]
        assert pcl.shape[1] == color.shape[1]

        mask_front = pcl[2, :] > 0
        z = pcl[2, :]
        pts_hm = pcl[0:3, :] / z
        pts_px = (np.round(np.matmul(self.K, pts_hm)))
        mask_zero = pts_px >= 0
        mask_width = pts_px[0] < self.width
        mask_height = pts_px[1] < self.height
        mask = mask_zero[0] * mask_zero[1] * mask_height * mask_width * mask_front
        u = pts_px[0][mask]
        v = pts_px[1][mask]

        r = color[0][mask]
        g = color[1][mask]
        b = color[2][mask]

        pts_px = np.vstack((u, v, np.ones((1, len(u)))))

        colormap = np.zeros((self.width * self.height, 3))

        idx = np.matmul(self.W, pts_px).astype(np.int)[0]

        colormap[idx, 0] = r
        colormap[idx, 1] = g
        colormap[idx, 2] = b

        return colormap.reshape((self.height, self.width, 3)).astype(format)

    def pclAtcamera(self, pcl, cameraPose_w2c):
        pcl_at_camera = np.dot(np.linalg.inv(cameraPose_w2c), pcl)
        return pcl_at_camera

    def undistort(self, image):
        return cv2.undistort(image, self.K, self.distortion_coeffs, None, self.Kp)

    def reprojectDepthImage(self, deph):
        normalized_coors = self.constructNormalizedCoordinate(deph)
        mask_grid = deph > 0
        # x_grid = normalized_coors[:, :, 0] * deph
        # y_grid = normalized_coors[:, :, 1] * deph

        pointcloud = normalized_coors * deph[:, :, None]
        pointcloud = np.dstack([pointcloud, deph])
        pointcloud = np.reshape(pointcloud, (-1, 3), order='F')
        mask_pcl = pointcloud[:, 2] > 0
        pointcloud = pointcloud[mask_pcl, :]
        pointcloud = np.hstack([pointcloud, np.ones((len(pointcloud[:, 0]), 1))])
        return pointcloud, mask_pcl, mask_grid

    def constructNormalizedCoordinate(self, array=None):
        # TODO this function is used in sphere camera projection by image2sphere. Maybe there are issues for that
        """
        Return normalized vectors [x, y, 1] from a generic image [h, w]
        define in the class
        """
        if array is None:
            h, w = self.get_shape()
        else:
            h, w = array.shape
        x = np.linspace(0, w - 1, w)
        y = np.linspace(0, h - 1, h)
        tilde_x, tilde_y = np.meshgrid(x, y)

        tilde_x = (tilde_x - self.cx) / self.fx
        tilde_y = (tilde_y - self.cy) / self.fy
        return np.dstack([tilde_x, tilde_y])

    def depthmap2pcl(self, depthmap, scaler=1, idx_matching=False, mask=None):

        # if mask is None:
        #     mask = depthmap > 0
        # else:
        #     mk = depthmap > 0
        #     mask = mask * mk

        mask = np.ones_like(depthmap).astype(np.bool)
        pts_px = np.array(np.where(mask))
        pts_px = np.vstack((pts_px[1, :], pts_px[0, :], np.ones((1, len(pts_px[1])))))

        pts_hm = np.matmul(self.Kinv, pts_px)
        pts_3D = scaler * np.reshape(depthmap, (1, -1), order='F') * pts_hm
        pts_3D = np.vstack((pts_3D, np.ones((1, len(pts_3D[1])))))
        mask_pcl = pts_3D[2, :] > 0
        pointcloud = pts_3D[:, mask_pcl]
        if idx_matching:
            idx = np.where(np.reshape(mask, (-1)))[0].astype(np.int)
            return pts_3D, idx

        return pointcloud

    def projectPointCloud(self, pcl):
        """
        Based on a PCL input (on camera frame) each point is projected on a image plane. This function outputs the
        pixel coordinates for each point in the pcl. This function is similar to pcl_projection #TODO merge with pcl_projection
        :param pcl: PCL [Nx3] or [Nx4]
        :return: pixels_points, indexes
        """
        # pcl to homogeneous coordinates
        hm_coords = pcl[0:3, :] / pcl[2, :]

        # projection using camera matrix
        img_coords = np.round(self.K.dot(hm_coords) - 0.5).astype(np.int)
        return img_coords

    @staticmethod
    def masking_pts(pcl):
        """
        This function computes a mask for all pts in front of a camera reference. This function is useful for 360 scene points, since
        it mask out all of the points with a z component lower than zero
        :param pcl: Points in 3D [n, 3], [n, 4]
        :return: mask
        """
        return pcl[:, 2] > 0

    def masking_coords(self, img_coords):
        """
        Given an array of pixels coordinates (3, N), a mask is computed for all of them which belong on a image plane
        defined for the intrinsic camera parameters K in the constructor
        :param img_coords: [3, N]  pixels coordinates
        :return: mask [3, N] boolean
        """
        # img_coords = img_coords - 0.5
        mask_zero = img_coords >= 0

        mask_width = img_coords[0, :] < self.width
        mask_height = img_coords[1, :] < self.height
        valid_mask = mask_zero[0, :] * mask_zero[1, :] * mask_height * mask_width
        return valid_mask

    def masking_idx(self, idx):
        unique, id, counts = np.unique(idx, return_counts=True, return_index=True)
        mask = np.zeros_like(idx).astype(np.bool)
        mask[id] = True
        return mask.reshape((-1), order='F')

    def warpimage(self, pcl, pose_rel, img_curr, return_mask=False, back_projection=True):

        assert img_curr.shape == self.get_shape()

        pcl_warped = np.dot(pcl, pose_rel.T)

        img_corrds, idx = self.projectPointCloud(pcl)
        img_corrds_warped, idx_warp = self.projectPointCloud(pcl_warped)

        # masking only coordinates in the range of width and height for the image
        mask_pcl = self.masking_coords(img_corrds)
        mask_warped_pcl = self.masking_coords(img_corrds_warped)

        # # masking for unique values. This part is not yet well implemented
        # mask_idx = self.masking_idx(idx)
        # mask_idx_warp = self.masking_idx(idx_warp)

        mask = mask_pcl * mask_warped_pcl
        # mask = mask_idx_warp * mask

        img_corrds = img_corrds[mask, :]
        img_corrds_warped = img_corrds_warped[mask, :]

        coords = [img_corrds, img_corrds_warped]

        img_warped = np.zeros_like(img_curr)
        # img_warped[img_corrds[:, 1], img_corrds[:, 0]] = img_curr[img_corrds_warped[:, 1], img_corrds_warped[:, 0]]
        if back_projection:
            img_warped[coords[0][:, 1], coords[0][:, 0]] = img_curr[coords[1][:, 1], coords[1][:, 0]]
        else:
            img_warped[coords[1][:, 1], coords[1][:, 0]] = img_curr[coords[0][:, 1], coords[0][:, 0]]

        if return_mask:
            return img_warped, mask, coords
        return img_warped

    def pxl2pcl(self, pcl, coord_pxl):
        """
        Based on pixels coordinates this function finds the correspondence pcl associated to it.
        First the pcl is projected using a camera definition, then the coordinates are query.
        :param pcl: point cloud array
        :param corrd_pxl: query coordinates
        :return: pcl associated to coord_pxl
        """
        # projecting PCL, return coord and idxs
        coord_img, idx = self.projectPointCloud(pcl)
        # mask for coord into an image frame [h, w]
        mask_pcl = self.masking_coords(coord_img)
        # masking coord
        coord_img = coord_img[mask_pcl, :]

        # PCL into an image frame
        pcl_projected = pcl[mask_pcl, :]
        coord_pxl = np.asarray(coord_pxl)

        pcl = []
        corrds = []
        for i in range(len(coord_pxl[0])):
            mask_u = coord_img[:, 0] == coord_pxl[0][i]
            mask_v = coord_img[:, 1] == coord_pxl[1][i]
            mask = mask_u * mask_v
            _pcl = pcl_projected[mask, :]
            dist = closest_pcl(_pcl)
            if np.max(mask) == False:
                continue
            else:
                pcl.append(_pcl[dist[0], :].reshape((1, 4)))
                corrds.append([coord_pxl[0][i], coord_pxl[1][i]])
            # print(index)
        return pcl, corrds

    def warping_pixels(self, pcl, pose_rel, img_curr, return_coords=False):
        assert pcl.__class__.__name__ == "PointCloud"

        pcl_warped = np.dot(pose_rel, pcl.pcl_)

        img_corrds, idx = self.projectPointCloud(pcl.pcl_.T)
        img_corrds_warped, idx_warp = self.projectPointCloud(pcl_warped.T)

        mask_pcl = self.masking_coords(img_corrds)
        mask_warped_pcl = self.masking_coords(img_corrds_warped)
        mask_idx = self.masking_idx(idx)
        mask_idx_warp = self.masking_idx(idx_warp)

        mask = mask_pcl * mask_warped_pcl
        # mask = mask_idx_warp * mask

        img_corrds = img_corrds[mask, :]
        img_corrds_warped = img_corrds_warped[mask, :]

        coords = [img_corrds, img_corrds_warped]

        img_warped = img_curr[coords[1][:, 1], coords[1][:, 0]]

        if return_coords:
            return img_warped, mask, coords
        else:
            return img_warped

    def pcl2image_plane(self, pcl):
        """
        Given a PCL (point cloud [N, 3] or [N, 4]) both a mask and an image are evaluated.
        The image represents the image projection defined by a K matrix camera in the constructor.
        The mask is the boolean array for the points PCL which are truly projected into a image plane.
        :param pcl: point cloud [N, 3] or [N, 4]
        :return: image [h, w] pcl projection
                 mask [N, 1] boolean mask for pcl projected on the image
        """
        mask_valid_pts = self.masking_pts(pcl)
        px_pts, idx = self.projectPointCloud(pcl=pcl)
        mask = self.masking_coords(px_pts) * mask_valid_pts
        image = np.zeros((self.get_shape())).reshape(-1)
        idx = idx[mask]
        image[idx] = 1
        return image.reshape(self.get_shape()), mask

    def pixels2homogeneous(self, pixels_coord):
        """
        Returns a set of points on homogeneous coordinates given an array of pixel coordinates (u, v)
        """
        assert pixels_coord.shape[1] in (2, 3), "{}".format(pixels_coord.shape)

        pixels_coord = np.hstack([pixels_coord[:, 0:2], np.ones((len(pixels_coord[:, 0]), 1))])

        return np.dot(pixels_coord, self.Kinv.T)

    def pinhole2sphere(self):
        """
        The spherical normalization is a normalization of the pixels points to homogeneous
        coordinates at first, then using a unit vector to a surface unit sphere.
        This units vectors are the pixels representation
        onto a sphere surface of ratio 1
        """
        hm_points = self.constructNormalizedCoordinate()
        hm_points = np.hstack([hm_points, np.ones((len(hm_points[:, 0]), 1))])
        norm = np.linalg.norm(hm_points, axis=1)
        return hm_points / norm

    @staticmethod
    def get_camera_from_yaml(calib):
        K = np.eye(3)
        K[0, 0] = calib["Camera.fx"]
        K[1, 1] = calib["Camera.fy"]
        K[0, 2] = calib["Camera.cx"]
        K[1, 2] = calib["Camera.cy"]

        dist = (calib["Camera.k1"],
                calib["Camera.k2"],
                calib["Camera.p1"],
                calib["Camera.p2"],
                calib["Camera.k3"])
        return K, dist, (calib["Camera.height"], calib["Camera.width"])


def build_image_pyramid(image_base, level=None):
    images_pyramid = []

    for i in range(level + 1):
        images_pyramid.append(image_base)
        image_base = cv2.pyrDown(image_base)

    if level is None:
        return images_pyramid
    else:
        return images_pyramid[level]


def closest_pcl(pcl):
    dist = np.argsort(np.linalg.norm(pcl[:, 0:3], axis=1))
    return dist


def plane_uvgrid(h, w):
    u = np.linspace(0, w - 1, num=w, dtype=np.float32)
    v = np.linspace(0, h - 1, num=h, dtype=np.float32)

    return np.stack(np.meshgrid(u, v), axis=-1)


def uv2homogenous(uv, camera):
    u, v = np.split(uv, 2, axis=-1)
    x = camera.Kinv[0, 0] * u + camera.Kinv[0, 2]
    y = camera.Kinv[1, 1] * v + camera.Kinv[1, 2]
    z = np.ones_like(x)

    return np.concatenate([x, y, z], axis=-1)
