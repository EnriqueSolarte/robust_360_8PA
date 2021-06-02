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
This script reads a registered pair of color and depth images and generates a colored 3D point cloud in PLY format.
"""

import argparse
import sys
import os
from associate import *
from evaluate_rpe import *
from generate_pointcloud import *
from PIL import Image
import struct

focalLength = 525.0
centerX = 319.5
centerY = 239.5
scalingFactor = 5000.0


def write_pcd(pcd_file, points, pose):
    file = open(pcd_file, "wb")
    file.write('''# .PCD v.7 - Point Cloud Data file format
VERSION .7
FIELDS x y z rgb
SIZE 4 4 4 4
TYPE F F F F
COUNT 1 1 1 1
WIDTH %d
HEIGHT 1
VIEWPOINT %f %f %f %f %f %f %f 
POINTS %d
DATA binary
%s
''' % (len(points), pose[0], pose[1], pose[2], pose[6], pose[3], pose[4],
       pose[5], len(points), "".join(points)))
    file.close()
    print("Saved %d points to '%s'" % (len(points), pcd_file))


def write_ply(ply_file, points):
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
    print("Saved %d points to '%s'" % (len(points), ply_file))


def plot_camera(cams, rvec, tvec, dim=(0.5, 1), vertices):
    ax = plt.gca(projection='3d')

    tvec = tvec.reshape((3, 1)).astype(np.float)
    rvec = rvec.reshape((3, 1)).astype(np.float)

    rmat = cv2.Rodrigues(rvec.astype(np.float))[0]

    if (len(oc) == 0):
        oc = c

    if len(dim) == 1:
        offset = float(dim[0])
        length = offset
    else:
        offset = float(dim[0])
        length = float(dim[1])

#    cam = np.tile(tvec, (1, 5))
    cam = np.zeros([3, 5])
    cam[0, 1:3] += offset
    cam[0, 3:] -= offset
    cam[1, 1::2] += offset
    cam[1, 2::2] -= offset
    cam[2, 1:] += 2.5 * length

    #    cam = rmat.dot(cam - tvec) + tvec
    cam = rmat.T.dot(cam - tvec)
    vertexNo = cams.shape[0] / 5
    #    ax.plot3D(cam[0, [0, 1]], cam[1, [0, 1]], cam[2, [0, 1]], color=c)
    #    ax.plot3D(cam[0, [0, 2]], cam[1, [0, 2]], cam[2, [0, 2]], color=c)
    #    ax.plot3D(cam[0, [0, 3]], cam[1, [0, 3]], cam[2, [0, 3]], color=c)
    #    ax.plot3D(cam[0, [0, 4]], cam[1, [0, 4]], cam[2, [0, 4]], color=c)
    #    ax.plot3D(cam[0, [1, 2]], cam[1, [1, 2]], cam[2, [1, 2]], color=c)
    #    ax.plot3D(cam[0, [2, 4]], cam[1, [2, 4]], cam[2, [2, 4]], color=c)
    #    ax.plot3D(cam[0, [3, 4]], cam[1, [3, 4]], cam[2, [3, 4]], color=c)
    #    ax.plot3D(cam[0, [3, 1]], cam[1, [3, 1]], cam[2, [3, 1]], color=c)
    for i in range(8):
        for j in range(i + 1, 8):
            vertices.append("%d %d %d %d %d" % (i, j, 255, 0, 0))

    for i in range(cam.shape[1]):
        cams.append("%f %f %f %d %d %d 0\n" %
                    (cam[0, i], vec_transf[1, i], vec_transf[2, i], 255, 0, 0))

    #return points.append("%f %f %f %d %d %d 0\n"%(vec_transf[0,0],vec_transf[1,0],vec_transf[2,0],255,0,0))


#    print 'cam', cam[0, 0], cam[1, 0], cam[2, 0]
#    ax.scatter(cam[0, 0], cam[1, 0], cam[2, 0], color=oc, marker='.')
#    ax.scatter(cam[0, 1::], cam[1, 1::], cam[2, 1::], color=c, marker='.')
#    ax.plot3D(cam[0, [0, 1]], cam[1, [0, 1]], cam[2, [0, 1]], color=c)
#    ax.plot3D(cam[0, [0, 2]], cam[1, [0, 2]], cam[2, [0, 2]], color=c)
#    ax.plot3D(cam[0, [0, 3]], cam[1, [0, 3]], cam[2, [0, 3]], color=c)
#    ax.plot3D(cam[0, [0, 4]], cam[1, [0, 4]], cam[2, [0, 4]], color=c)
#    ax.plot3D(cam[0, [1, 2]], cam[1, [1, 2]], cam[2, [1, 2]], color=c)
#    ax.plot3D(cam[0, [2, 4]], cam[1, [2, 4]], cam[2, [2, 4]], color=c)
#    ax.plot3D(cam[0, [3, 4]], cam[1, [3, 4]], cam[2, [3, 4]], color=c)
#    ax.plot3D(cam[0, [3, 1]], cam[1, [3, 1]], cam[2, [3, 1]], color=c)


def generate_pointcloudWithCamera(rgb_file,
                                  depth_file,
                                  transform,
                                  downsample,
                                  pcd=False):
    """
    Generate a colored point cloud 
    
    Input:
    rgb_file -- filename of color image
    depth_file -- filename of depth image
    transform -- camera pose, specified as a 4x4 homogeneous matrix
    downsample -- downsample point cloud in x/y direction
    pcd -- true: output in (binary) PCD format
           false: output in (text) PLY format
           
    Output:
    list of colored points (either in binary or text format, see pcd flag)
    """

    rgb = Image.open(rgb_file)
    depth = Image.open(depth_file)

    if rgb.size != depth.size:
        raise Exception(
            "Color and depth image do not have the same resolution.")
    if rgb.mode != "RGB":
        raise Exception("Color image is not in RGB format")
    if depth.mode != "I":
        raise Exception("Depth image is not in intensity format")

    points = []
    for v in range(0, rgb.size[1], downsample):
        for u in range(0, rgb.size[0], downsample):
            color = rgb.getpixel((u, v))
            Z = depth.getpixel((u, v)) / scalingFactor
            if Z == 0: continue
            X = (u - centerX) * Z / focalLength
            Y = (v - centerY) * Z / focalLength
            vec_org = numpy.matrix([[X], [Y], [Z], [1]])
            if pcd:
                points.append(
                    struct.pack(
                        "fffI", vec_org[0, 0], vec_org[1, 0], vec_org[2, 0],
                        color[0] * 2**16 + color[1] * 2**8 + color[2] * 2**0))
            else:
                vec_transf = numpy.dot(transform, vec_org)
                points.append("%f %f %f %d %d %d 0\n" %
                              (vec_transf[0, 0], vec_transf[1, 0],
                               vec_transf[2, 0], color[0], color[1], color[2]))

    return points


def generate_pointcloud(rgb_file,
                        depth_file,
                        transform,
                        downsample,
                        pcd=False):
    """
    Generate a colored point cloud 
    
    Input:
    rgb_file -- filename of color image
    depth_file -- filename of depth image
    transform -- camera pose, specified as a 4x4 homogeneous matrix
    downsample -- downsample point cloud in x/y direction
    pcd -- true: output in (binary) PCD format
           false: output in (text) PLY format
           
    Output:
    list of colored points (either in binary or text format, see pcd flag)
    """

    rgb = Image.open(rgb_file)
    depth = Image.open(depth_file)

    if rgb.size != depth.size:
        raise Exception(
            "Color and depth image do not have the same resolution.")
    if rgb.mode != "RGB":
        raise Exception("Color image is not in RGB format")
    if depth.mode != "I":
        raise Exception("Depth image is not in intensity format")

    points = []
    for v in range(0, rgb.size[1], downsample):
        for u in range(0, rgb.size[0], downsample):
            color = rgb.getpixel((u, v))
            Z = depth.getpixel((u, v)) / scalingFactor
            if Z == 0: continue
            X = (u - centerX) * Z / focalLength
            Y = (v - centerY) * Z / focalLength
            vec_org = numpy.matrix([[X], [Y], [Z], [1]])
            if pcd:
                points.append(
                    struct.pack(
                        "fffI", vec_org[0, 0], vec_org[1, 0], vec_org[2, 0],
                        color[0] * 2**16 + color[1] * 2**8 + color[2] * 2**0))
            else:
                vec_transf = numpy.dot(transform, vec_org)
                points.append("%f %f %f %d %d %d 0\n" %
                              (vec_transf[0, 0], vec_transf[1, 0],
                               vec_transf[2, 0], color[0], color[1], color[2]))

    return points


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='''
    This script reads a registered pair of color and depth images and generates a colored 3D point cloud in the
    PLY format. 
    ''')
    parser.add_argument('rgb_list',
                        help='input color image (format: timestamp filename)')
    parser.add_argument('depth_list',
                        help='input depth image (format: timestamp filename)')
    parser.add_argument(
        'trajectory_file',
        help='input trajectory (format: timestamp tx ty tz qx qy qz qw)')
    parser.add_argument(
        '--depth_offset',
        help=
        'time offset added to the timestamps of the depth file (default: 0.00)',
        default=0.00)
    parser.add_argument(
        '--depth_max_difference',
        help=
        'maximally allowed time difference for matching rgb and depth entries (default: 0.02)',
        default=0.02)
    parser.add_argument(
        '--traj_offset',
        help=
        'time offset added to the timestamps of the trajectory file (default: 0.00)',
        default=0.00)
    parser.add_argument(
        '--traj_max_difference',
        help=
        'maximally allowed time difference for matching rgb and traj entries (default: 0.01)',
        default=0.02)
    parser.add_argument('--downsample',
                        help='downsample images by this factor (default: 1)',
                        default=1)
    parser.add_argument('--nth',
                        help='only consider every nth image pair (default: 1)',
                        default=1)
    parser.add_argument(
        '--individual',
        help='save individual point clouds (instead of one large point cloud)',
        action='store_true')
    parser.add_argument(
        '--pcd_format',
        help='Write pointclouds in pcd format (implies --individual)',
        action='store_true')

    parser.add_argument('output_file', help='output PLY file (format: ply)')
    args = parser.parse_args()

    rgb_list = read_file_list(args.rgb_list)
    depth_list = read_file_list(args.depth_list)
    pose_list = read_file_list(args.trajectory_file)
    #print np.shape(rgb_list),np.shape(depth_list),np.shape(pose_list)
    matches_rgb_depth = dict(
        associate(rgb_list, depth_list, float(args.depth_offset),
                  float(args.depth_max_difference)))
    matches_rgb_traj = associate(matches_rgb_depth, pose_list,
                                 float(args.traj_offset),
                                 float(args.traj_max_difference))
    matches_rgb_traj.sort()
    #print np.shape(matches_rgb_traj)
    if args.pcd_format:
        args.individual = True
        traj = read_trajectory(args.trajectory_file, False)
    else:
        traj = read_trajectory(args.trajectory_file)

    all_points = []
    list = list(range(0, len(matches_rgb_traj), int(args.nth)))
    for frame, i in enumerate(list):
        rgb_stamp, traj_stamp = matches_rgb_traj[i]

        if args.individual:
            if args.pcd_format:
                out_filename = "%s-%f.pcd" % (os.path.splitext(
                    args.output_file)[0], rgb_stamp)
            else:
                out_filename = "%s-%f.ply" % (os.path.splitext(
                    args.output_file)[0], rgb_stamp)
            if os.path.exists(out_filename):
                print("skipping existing cloud file ", out_filename)
                continue

        rgb_file = rgb_list[rgb_stamp][0]
        depth_file = depth_list[matches_rgb_depth[rgb_stamp]][0]
        pose = traj[traj_stamp]
        points = generate_pointcloud(rgb_file, depth_file, pose,
                                     int(args.downsample), args.pcd_format)

        if args.individual:
            if args.pcd_format:
                write_pcd(out_filename, points, pose)
            else:
                write_ply(out_filename, points)
        else:
            all_points += points
            print("Frame %d/%d, number of points so far: %d" %
                  (frame + 1, len(list), len(all_points)))
        if (frame + 1 > 50):
            break

    if not args.individual:
        write_ply(args.output_file, all_points)
