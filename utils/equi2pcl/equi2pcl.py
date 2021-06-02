from utils.camera_models.sphere import Sphere
from imageio import imread
from utils.vispy_utilities import plot_color_plc
import numpy as np
from utils.image_utilities import get_shape


class Equi2PCL:
    def __init__(self, shape):
        self.cam = Sphere(shape=shape)
        self.scaler = 1

    def plot2sphere(self, color_map):
        # ! Assert the color_map shape top the shaper defined in self.cam
        h, w = get_shape(color_map)
        assert (h, w) == self.cam.shape
        color_pixels = self.cam.get_color_array(color_map=color_map) / 255
        plot_color_plc(self.cam.default_bearings.T, color_pixels.T)

    def plot2pcl(self, color_map, depth_map):
        pcl, color_pcl = self.get_pcl(color_map, depth_map)
        plot_color_plc(pcl.T, color_pcl.T)

    def get_pcl(self, color_map, depth_map, scaler=1):
        h, w = get_shape(color_map)
        assert (h, w) == depth_map.shape, "color frame must be == to depth map"
        color_pixels = self.cam.get_color_array(color_map=color_map) / 255
        mask = depth_map.flatten() > 0
        pcl = self.cam.default_bearings[:, mask] * (
            1 / scaler) * depth_map.flatten()[mask]
        return pcl, color_pixels[mask, :]


if __name__ == '__main__':
    color_file = "../data/sample_images/scene_0/0/rgb/1.png"
    depth_file = "../data/sample_images/scene_0/0/depth/1.png"

    color_ = imread(color_file)
    depth_ = imread(depth_file)

    e2pcl = Equi2PCL(shape=depth_.shape)
    e2pcl.plot2sphere(color_map=color_)
    e2pcl.plot2pcl(color_map=color_, depth_map=depth_)
    print('done')
