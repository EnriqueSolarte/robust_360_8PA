#!/usr/bin/python
# Software License Agreement (BSD License)
#
# Copyright (c) 2013, Juergen Sturm, TUM
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above
#    copyright notice, this list of conditions and the following
#    disclaimer in the documentation and/or other materials provided
#    with the distribution.
#  * Neither the name of TUM nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
# COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
# ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
#
# the resulting .ply file can be viewed for examples with meshlab
# sudo apt-get install meshlab
"""
This script reads a registered pair of color and depth images and generates a
colored 3D point cloud in the PLY format.
"""

import argparse
import sys
import os
import cv2
import numpy as np
from PIL import Image

#fx = fy = 525.0
#centerX = 319.5
#centerY = 239.5

fx = 517.3
fy = 516.5
centerX = 318.6
centerY = 255.3
scalingFactor = 5000.0


def generate_pointcloud(rgb_file, depth_file, ply_file):
    """
    Generate a colored point cloud in PLY format from a color and a depth image.
    
    Input:
    rgb_file -- filename of color image
    depth_file -- filename of depth image
    ply_file -- filename of ply file
    
    """
    rgb = cv2.imread(rgb_file)  #Image.open(rgb_file)

    depth = Image.open(depth_file)

    if rgb.size != depth.size:
        raise Exception(
            "Color and depth image do not have the same resolution.")
    if rgb.mode != "RGB":
        raise Exception("Color image is not in RGB format")
    if depth.mode != "I":
        raise Exception("Depth image is not in intensity format")

    K = np.matrix([[fx, 0, centerX], [0, fy, centerY], [0, 0, 1]])
    distC = np.array([0.2624, -0.9531, -0.0054, 0.0026, 1.1633])
    rgbU = cv2.undistort(rgb, K, distC)
    points = []
    for v in range(rgb.size[1]):
        for u in range(rgb.size[0]):
            color = rgbU.getpixel((u, v))
            Z = depth.getpixel((u, v)) / scalingFactor
            if Z == 0: continue
            X = (u - centerX) * Z / fx
            Y = (v - centerY) * Z / fy
            points.append("%f %f %f %d %d %d 0\n" %
                          (X, Y, Z, color[0], color[1], color[2]))
    file = open(ply_file, "w")
    file.write('''ply
format ascii 1.0
element vertex %d
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
property uchar alpha
end_header
%s
''' % (len(points), "".join(points)))
    file.close()


doUndistort = True
if __name__ == '__main__':
    #    generate_pointcloud2("/datawl/CSI/CSI_SVN/Kiras CSISmartCam3D/CSISmartScan3D_Daten/Daten_Testszenen/TUM/freiburg1/rgbd_dataset_freiburg1_xyz/rgb/1305031102.175304.png",
    #                        "/datawl/CSI/CSI_SVN/Kiras CSISmartCam3D/CSISmartScan3D_Daten/Daten_Testszenen/TUM/freiburg1/rgbd_dataset_freiburg1_xyz/depth/1305031102.160407.png",
    #                        "./test_new.ply")
    rgb = cv2.imread(
        "/datawl/CSI/CSI_SVN/Kiras CSISmartCam3D/CSISmartScan3D_Daten/Daten_Testszenen/TUM/freiburg1/rgbd_dataset_freiburg1_xyz/rgb/1305031102.175304.png"
    )
    depth = cv2.imread(
        "/datawl/CSI/CSI_SVN/Kiras CSISmartCam3D/CSISmartScan3D_Daten/Daten_Testszenen/TUM/freiburg1/rgbd_dataset_freiburg1_xyz/depth/1305031102.160407.png",
        cv2.CV_LOAD_IMAGE_UNCHANGED)
    K = np.matrix([[fx, 0, centerX], [0, fy, centerY], [0, 0, 1]])
    distC = np.array([0.2624, -0.9531, -0.0054, 0.0026, 1.1633])
    newCameraMatrix = np.empty([3, 3])
    if (doUndistort):
        rgbU = cv2.undistort(rgb, K, distC, None, K)
    else:
        rgbU = rgb
    points = []
    for v in range(rgb.shape[0]):
        for u in range(rgb.shape[1]):
            color = rgbU[v, u, :]
            Z = depth[v, u] / scalingFactor
            if Z == 0: continue
            X = (u - centerX) * Z / fx
            Y = (v - centerY) * Z / fy
            points.append("%f %f %f %d %d %d 0\n" %
                          (X, Y, Z, color[2], color[1], color[0]))
    file = open("./test_new.ply", "w")
    file.write('''ply
format ascii 1.0
element vertex %d
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
property uchar alpha
end_header
%s
''' % (len(points), "".join(points)))
    file.close()

#    parser = argparse.ArgumentParser(description='''
#    This script reads a registered pair of color and depth images and generates a colored 3D point cloud in the
#    PLY format.
#    ''')
#    parser.add_argument('rgb_file', help='input color image (format: png)')
#    parser.add_argument('depth_file', help='input depth image (format: png)')
#    parser.add_argument('ply_file', help='output PLY file (format: ply)')
#    args = parser.parse_args()
#
#    generate_pointcloud(args.rgb_file,args.depth_file,args.ply_file)
