"""
The goal of this script is to visualize what are the error_surfaces for a particular pair of
frames (Kf-frm) by using both RANSAC and without RANSAC (i.e., with outliers and with inliers only)
"""
from analysis.utilities import *
from image_utilities import get_mask_map_by_res_loc
from read_datasets.MP3D_VO import MP3D_VO
from solvers.epipolar_constraint_by_ransac import RansacEssentialMatrix
from solvers.epipolar_constraint import EightPointAlgorithmGeneralGeometry
from structures.tracker import LKTracker
from structures.extractor.shi_tomasi_extractor import Shi_Tomasi_Extractor
from solvers.optimal8pa import Optimal8PA as norm_8pa
from geometry_utilities import *
from analysis.error_loss.error_surfaces_8PA import plot_surfaces
import os


def load_results(**kwargs):
    kwargs.update(load_obj(kwargs["filename"] + ".data"))
    plot_surfaces(**kwargs)
    print("done")


if __name__ == '__main__':
    # data = "plots/surface_2azQ1b91cZZ_res_180.180_dist0.5_kf220_frm228_grid_-1.1.10_projected_distance"
    data = "plots/surface_2azQ1b91cZZ_res_360.180_dist0.5_kf85_frm91_grid_-1.1.50_projected_distance"
    model_settings = dict(
        filename=data,
        # mask_results=('inliers_pts_error_residual', "inliers_pts_error_rot",
        #               "inliers_pts_error_tran", "inliers_pts_error_e",
        #               'all_pts_error_e'),
        # mask_quantile=0.5,
    )

    load_results(**model_settings)
