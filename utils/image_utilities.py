import cv2
import numpy as np
from scipy import ndimage
from scipy.interpolate import interp2d
from utils.file_utils import create_dir
import os
import matplotlib.pyplot as plt


def merge_images(images):
    msk = np.ones_like(images[0][0])
    image = np.zeros_like(msk)
    for img, mask in images:
        image[msk>0] = img[msk>0]
        msk[mask>0]=0
        # cv2.imshow("image", image)
        # cv2.waitKey(0)
    return image


def get_cmap(n, name='hsv'):
    '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct 
    RGB color; the keyword argument name must be a standard mpl colormap name.'''
    return plt.cm.get_cmap(name, n)


def get_shape(array_map):
    """
     Useful function to get the shape of an image array regardless
     of its shape.
     :return (h, w)
    """
    if len(array_map.shape) > 2:
        h, w, c = array_map.shape
    else:
        h, w = array_map.shape
    return h, w


def get_max_gradients(image, ksize=5, gaussian=0.2, threshold=0.2):
    image = cv2.GaussianBlur(image, (3, 3), 0)
    sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=ksize)
    sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=ksize)
    mag = np.hypot(sobelx, sobely)
    mag *= 255.0 / np.max(mag)
    mag[mag < threshold * np.max(mag)] = 0
    return np.array(mag)


def interpolate(array, shape, type=np.float64):
    return np.interp(array, (array.min(), array.max()), shape).astype(type)


def gradient_cv(image, mask=True, size=3, threshold=0.01):
    # image = cv2.GaussianBlur(image, (3, 3), 0)
    sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=size)
    sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=size)
    mag = np.sqrt(sobelx ** 2 + sobely ** 2)
    max_grad = np.max(mag)

    mk = mag < threshold * max_grad
    if mask:
        mag[mk] = 0
        sobelx[mk] = 0
        sobely[mk] = 0

    # sobelx = np.nan_to_num(sobelx / max(sobelx))
    # sobely = np.nan_to_num(sobely / max(sobely))
    # mag *= (1 / np.max(mag))
    # sobelx = normlizer(sobelx)
    # sobely = normlizer(sobely)
    return [sobelx, sobely, mag]


def normlizer(array):
    m = 2 / (array.max() - array.min())
    b = 1 - m * array.max()

    return m * array + b


def gradient_numpy(array, size=1, mask=True, threshold=0.01):
    grad_y = np.gradient(array, size, edge_order=2, axis=0)
    grad_x = np.gradient(array, size, edge_order=2, axis=1)
    mag = np.sqrt(grad_x ** 2 + grad_y ** 2)
    max_grad = np.max(mag)

    mk = mag < threshold * max_grad
    if mask:
        mag[mk] = 0
        grad_x[mk] = 0
        grad_y[mk] = 0

    return [grad_x, grad_y, mag]


def gradient_scipy(array):
    from scipy import ndimage
    # Get x-gradient in "sx"
    sy = ndimage.sobel(array, axis=0, mode='nearest')
    # Get y-gradient in "sy"
    sx = ndimage.sobel(array, axis=1, mode='nearest')

    return [sx, sy]


def get_gradient_vetors(image,
                        ksize=5,
                        threshold=0,
                        mask=True,
                        stack=False,
                        all=False,
                        mode=0):
    image = cv2.GaussianBlur(image, (25, 25), 0)
    sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=ksize)
    sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=ksize)

    mag = np.sqrt(sobelx ** 2 + sobely ** 2)
    max_grad = np.max(mag)
    mk = mag < threshold * max_grad
    if mask:
        sobelx[mk] = 0
        sobely[mk] = 0
        mag[mk] = 0

    sobelx = np.nan_to_num(sobelx / max_grad)
    sobely = np.nan_to_num(sobely / max_grad)
    mag *= (1 / np.max(mag))

    h, w = image.shape
    if stack:
        if not all:
            image_gradient = np.zeros((h, w, 3))
            image_gradient[:, :, 0] = sobelx
            image_gradient[:, :, 1] = sobely
            image_gradient[:, :, 2] = mag
            return image_gradient
        else:
            image_gradient = np.zeros((h, w, 4))
            image_gradient[:, :, 0] = sobelx
            image_gradient[:, :, 1] = sobely
            image_gradient[:, :, 2] = mag
            image_gradient[:, :, 3] = mag > threshold * np.max(mag)
            return image_gradient

    if mode is 1:
        return sobelx
    elif mode is 2:
        return sobely
    elif mode is 3:
        return mag
    elif mode is 123:
        return sobelx, sobely, mag
    elif mode is 12:
        return sobelx, sobely

    return sobelx, sobely, mag, mag > threshold * np.max(mag)


def get_gradient_pixels(image_grad):
    return np.where(image_grad > 0)


def get_combined_image(image_i, image_ipp, type="gradient"):
    assert image_i.shape == image_ipp.shape
    combined_image = -1
    if type is "gradient":
        combined_image = np.zeros((image_ipp.shape[0], image_ipp.shape[1], 3))
        image_i = cv2.cvtColor(image_i, cv2.COLOR_BGR2GRAY)
        image_ipp = cv2.cvtColor(image_ipp, cv2.COLOR_BGR2GRAY)
        combined_image[:, :, 0] = get_max_gradients(image_i)
        combined_image[:, :, 1] = get_max_gradients(image_ipp)

    if type is "gray_residual":
        if image_ipp.shape[2] is 3:
            image_i = cv2.cvtColor(image_i, cv2.COLOR_BGR2GRAY)
            image_ipp = cv2.cvtColor(image_ipp, cv2.COLOR_BGR2GRAY)
        combined_image = image_i - image_ipp

    if type is "color_residual":
        combined_image = image_i - image_ipp

    if type is "adding":
        combined_image = image_i + image_ipp

    return combined_image


def get_gray_scale_image(image):
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return img_gray


def scale_to_255(image):
    max = np.max(image)
    min = np.min(image)
    scl = 255 * (image - min) / (max - min)
    return scl.astype(np.uint8)


def get_uv_plane2grid(h, w):
    u = np.linspace(0, w, w + 1, dtype=np.float32)
    v = np.linspace(0, h, h + 1, dtype=np.float32)
    uu, vv = np.meshgrid(u, v)
    return np.vstack((vv.flatten(), uu.flatten()))


def get_theta_phi_grid(h, w):
    """
    from a shape(h, w) returns a grid of spherical coords (theta, phi)
    """
    from geometry_utilities import extend_array_to_homogeneous as ext

    pixels = get_uv_plane2grid(h, w)
    Kinv = np.asarray((2 * np.pi / w, 0, -np.pi, 0, -np.pi / h, np.pi / 2, 0,
                       0, 1)).reshape(3, 3)

    pixels = ext(pixels)
    return Kinv.dot(pixels)


def apply_on_image(image, patch, idx, function):
    """
    It applies the passed function to a group of pixels in a image
    given a particular patch (kernel size) and the center locations
    """
    patch = patch[0] // 2, patch[1] // 2
    for (v, u) in zip(idx[0], idx[1]):
        if (v + patch[0] > image.shape[0]) or (v - patch[0]) < 0:
            continue
        if (u + patch[1] > image.shape[1]) or (u - patch[1]) < 0:
            continue
        image[v - patch[0]:v + patch[0], u - patch[1]:u + patch[1]] = function(
            image[v - patch[0]:v + patch[0], u - patch[1]:u + patch[1]])

    return image


def split_mask(mask):
    masks = []
    h, w = mask.shape

    auxiliary_mask = np.zeros_like(mask)
    auxiliary_mask[0:h // 2, 0:w // 2] = 255
    masks.append(auxiliary_mask)

    auxiliary_mask = np.zeros_like(mask)
    auxiliary_mask[h // 2:-1, 0:w // 2] = 255
    masks.append(auxiliary_mask)

    auxiliary_mask = np.zeros_like(mask)
    auxiliary_mask[0:h // 2, w // 2:-1] = 255
    masks.append(auxiliary_mask)

    auxiliary_mask = np.zeros_like(mask)
    auxiliary_mask[h // 2:-1, w // 2:-1] = 255
    masks.append(auxiliary_mask)

    return masks


def get_mask_map_by_close_zero_pixels(image):
    '''
    Returns a mask mao given an image sample. This mask mask-in all pixles values greater than zero
    '''
    passd


def get_mask_map_by_res_loc(shape, res, loc):
    """
    returns a mask map given a resolution res=(theta, phi) and location
    loc(theta, phi) camera orientation
    """
    assert len(shape) == 2
    from geometry_utilities import extend_array_to_homogeneous as ext

    h, w = shape
    theta = (-res[0] / 2 + loc[0]), res[0] / 2 + loc[0]
    phi = -res[1] / 2 + loc[1], res[1] / 2 + loc[1]
    # ! (theta, phi) = Kinv * (u, v)
    K = np.linalg.inv(
        np.asarray((2 * np.pi / w, 0, -np.pi, 0, -np.pi / h, np.pi / 2, 0, 0,
                    1)).reshape(3, 3))
    sph_coord = np.radians(np.vstack((theta, phi)))
    uv_coord = K.dot(ext(sph_coord)).astype(int)
    mask = np.zeros((h, w))
    mask[uv_coord[1, 1]:uv_coord[1, 0], uv_coord[0, 0]:uv_coord[0, 1]] = 1
    return (mask * 255).astype(np.uint8)


def uv_plane2homogenous(uv, camera):
    u, v = np.split(uv, 2, axis=-1)
    x = camera.Kinv[0, 0] * u + camera.Kinv[0, 2]
    y = camera.Kinv[1, 1] * v + camera.Kinv[1, 2]
    z = np.ones_like(x)

    return np.concatenate([x, y, z], axis=-1)


def interpolate2d(image, x, y, kind='linear'):
    f = interp2d(x, y, image, kind=kind)


def get_indexes(mask):
    h, w = mask.shape
    pts_px = np.array(np.where(mask))

    idx = pts_px[0] * w + pts_px[1]

    return idx


def get_corners_positions(image, threshold=0.1, corners=1):
    assert np.max(image) > 1, "Only non-normalized images"

    gray = np.float32(image)
    dst = cv2.cornerHarris(gray, 2, 3, 0.04)

    # result is dilated for marking the corners, not important
    # dst = cv2.dilate(dst, None)
    pos = np.where(dst > threshold * dst.max())

    return [pos[1], pos[0]]


def get_gradients_positions(image, threshold=0.1):
    img_grad_list = gradient_numpy(image,
                                   size=1,
                                   mask=True,
                                   threshold=threshold)

    pos = np.where(img_grad_list[2] > 0)

    return [pos[1], pos[0]]


def get_pixels_positions(image, threshold=0):
    pos = np.where(image > threshold)

    return [pos[1], pos[0]]


def draw_bbox_on_image(image, pos, only_box=False):
    if len(image.shape) < 3:
        h, w = image.shape

        if only_box:
            img = image[pos[0][0]:pos[1][0], pos[0][1]:pos[1][1]]
            return img
        else:
            img = np.zeros((h, w, 3))

            img[:, :, 0] = image
            img[:, :, 1] = image
            img[:, :, 2] = image

    else:
        if only_box:
            img = image[pos[0][0]:pos[1][0], pos[0][1]:pos[1][1], :]
            return img
        else:

            img = image

    cv2.rectangle(img,
                  tuple(pos[0]),
                  tuple(pos[1]), (0, 255, 0),
                  thickness=1,
                  lineType=1,
                  shift=0)

    return img


def masking_image(image, mask=None, inner=True):
    """
    Given a boolean mask [m x n]  only the inner or outer values of an image filtered out
    :param image: Image to be filtered [m x n]
    :param mask: Mask image [m x n] True: inner, False: Outer
    :param inside: Flag for inner or outer as output
    :return: image filtered by the masking [m x n]
    """

    img = np.zeros_like(image)
    if inner:
        img[mask] = image[mask]
    else:
        img[~mask] = image[~mask]

    return img


def show_images(images,
                _label="image",
                extra_info="",
                wait_for=0,
                write_on_image=True,
                save=False,
                display_mode=0,
                folder=None,
                show=True):
    """
    Convenient function for plotting images quickly
    :param extra_info: extra information to plots in the image
    :param _label: Label title for showing
    :param images: List of images array list([h, w, c], [h, w, c], [h, w, c], ....
    :return:  1
    """

    for idx, img in enumerate(images):
        if len(img.shape) > 2:
            h, w, c = img.shape
        else:
            h, w = img.shape
        winname = "{} {} {}".format(_label, idx, extra_info)
        if write_on_image:
            img = (img * 255).astype(np.uint8)
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(img, winname, (10, 15), font, 0.5, (255, 255, 255), 1,
                        cv2.LINE_AA)

        if show:
            cv2.namedWindow(winname)

            if display_mode is 0:
                cv2.moveWindow(winname, int(40 + 00.5 * w * idx),
                               int(30 + 0.5 * h * idx))
            if display_mode is 1:
                cv2.moveWindow(winname, int(40 + w * idx), int(30))
            if display_mode is 2:
                cv2.moveWindow(winname, int(40), int(30 + h * idx))

        if show:
            cv2.imshow(winname, img)

        if save:
            fname = "{}_{}_{}".format(_label, idx, extra_info)
            if folder is not None:
                if not os.path.exists(folder):
                    os.makedirs(folder)
                filename = folder + "/{}.png".format(fname)
            else:
                filename = "{}.png".format(fname)

            cv2.imwrite(filename, img)

    if show:
        cv2.waitKey(wait_for)
        cv2.destroyAllWindows()
    return 1


def plot_key_points_in_image(image, key_points, color=(0, 255, 0), size=2):
    assert key_points.shape[1] == 2
    for u, v in zip(key_points[:, 1], key_points[:, 0]):
        cv2.circle(image, (v, u), size, color, -1)


def write_on_image(image, label, loc=(50, 50), plot=True):
    img = cv2.putText(image, label, (50, 50), cv2.FONT_HERSHEY_SIMPLEX,
                      1, (255, 0, 0), 2, cv2.LINE_AA)
    if plot:
        cv2.imshow("frm", img)


def get_color_list(array_colors=None, fr=0.1, return_list=False, number_of_colors=None):
    """
    Returs a different color RGB for every element in the array_color
    """
    if array_colors is not None:
        number_of_colors = len(array_colors)

    values = np.linspace(0, np.pi, number_of_colors)
    colors = np.ones((3, number_of_colors))

    colors[0, :] *= abs(np.sin(values * (number_of_colors) * np.pi))
    colors[1, :] *= abs(np.cos(values * (number_of_colors*2) * np.pi))
    # colors[2, :] *= abs(np.sin(values * (number_of_colors*1000) * np.pi))
    colors[2, :] = 1-colors[0, :]
    if return_list:
        return [c for c in colors.T]
    return colors
