# -*- coding: utf-8 -*-
"""
Created on Tue Aug  9 08:33:31 2016

@author: fschenk
"""

import cv2
import cv
import numpy as np
import matplotlib.pyplot as plt

#test = []
#
#res = cv.Load('/datawl/CSI/CSI_SVN/Kiras CSISmartCam3D/CSISmartScan3D_Daten/Software/InfiniTAM-build/my.xml')
#pcl = cv.Load('/datawl/CSI/CSI_SVN/Kiras CSISmartCam3D/CSISmartScan3D_Daten/Software/InfiniTAM-build/pcl.xml')
##res = cv2.imread('/datawl/CSI/CSI_SVN/Kiras CSISmartCam3D/CSISmartScan3D_Daten/Daten_Testszenen/TUM/raycast_image.exr');
#fx = 525.0
#fy =  525.0
#cx = 319.5
#cy =  239.5
#K = np.zeros([3,3])
#K[0,0] = fx; K[1,1] = fy;
#K[0,2] = cx; K[1,2] = cy;
#K[2,2] = 1;
#res = pcl
#d = np.zeros([480,640])
#for xx in range(640):
#    for yy in range(480):
#        _2d = K.dot(res[yy,xx][0:3])
#        x = _2d[0]/_2d[2]
#        y = _2d[1]/_2d[2]
#        if (x >= 0 and x < 640 and y >= 0 and y < 480):
#            d[y,x] = res[yy,xx][2];
#d2 = d*1.0/0.005
#plt.imshow(d2)
#stop


def unprojectPtsWithDepth(edges, depth):
    nPoints = 0
    _3dedges = []
    #points = []
    for xx in range(edges.shape[1]):
        for yy in range(edges.shape[0]):
            if (edges[yy, xx] > 0 and depth[yy, xx] > 0.1):
                nPoints += 1
                Z = depth[yy, xx]
                X = Z * (xx - cx) / fx
                Y = Z * (yy - cy) / fy
                _3dedges.append([X, Y, Z])
                #print X,Y,Z
                #color = rgb[yy,xx];
                #points.append("%f %f %f %d %d %d 0\n"%(X,Y,Z,color[0],color[1],color[2]))
    #print nPoints
    #file = open("test_ply.ply","w")
    #file.write('''ply
    #format ascii 1.0
    #element vertex %d
    #property float x
    #property float y
    #property float z
    #property uchar red
    #property uchar green
    #property uchar blue
    #property uchar alpha
    #end_header
    #%s
    #'''%(len(points),"".join(points)))
    #file.close()
    return _3dedges


def transform3DPcl(_3dedges, R, t, K):
    _3dedges = np.matrix(_3dedges).transpose()
    _3dedges_transf = R.T * (_3dedges - t.reshape([3, 1]))
    #_3dedges_transf = (R*_3dedges+t.reshape([3,1]))
    _3dedges_transf /= _3dedges_transf[2, :]

    return K * _3dedges_transf


def generateEdgeImg(_2dreproj, shape):
    newEdges = np.zeros(shape, dtype='uint8')

    for i in range(_2dreproj.shape[1]):
        x = (np.floor(_2dreproj[0, i]))
        y = (np.floor(_2dreproj[1, i]))
        #print _3dedges_transf[0,index],_3dedges_transf[1,index],x,y
        if (x >= 0 and y >= 0 and x < newEdges.shape[1]
                and y < newEdges.shape[0]):
            newEdges[y, x] = 255
    return newEdges


rgb = cv2.imread(
    '/datawl/CSI/CSI_SVN/Kiras CSISmartCam3D/CSISmartScan3D_Daten/Daten_Testszenen/TUM/freiburg1/rgbd_dataset_freiburg1_desk/rgb/1305031453.359684.png'
)
depth = cv2.imread(
    '/datawl/CSI/CSI_SVN/Kiras CSISmartCam3D/CSISmartScan3D_Daten/Daten_Testszenen/TUM/freiburg1/rgbd_dataset_freiburg1_desk/depth/1305031453.374112.png',
    cv2.IMREAD_UNCHANGED)
depth = depth / 5000.0
gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
edges = np.array([])
edges = cv2.Canny(gray, 150, 100, edges, 3, True)

fx = 525.0
fy = 525.0
cx = 319.5
cy = 239.5
K = np.zeros([3, 3])
K[0, 0] = fx
K[1, 1] = fy
K[0, 2] = cx
K[1, 2] = cy
K[2, 2] = 1

_3dedges = unprojectPtsWithDepth(edges, depth)
R = np.matrix([[0.999693, -0.0220359, 0.0113385],
               [0.022001, 0.999753, 0.00319164],
               [-0.0114061, -0.0029412, 0.99993]])
t = np.array([0.00271005, 0.0022586, -0.00904049])
_2dreproj1 = transform3DPcl(_3dedges, R, t, K)
newEdges = generateEdgeImg(_2dreproj1, edges.shape[:])

R = np.matrix([[0.999698, -0.0223397, 0.0102584],
               [0.0223036, 0.999745, 0.00361705],
               [-0.0103366, -0.00338715, 0.999941]])
t = np.array([0.01400186, 0.0209673, -0.089684])
_2dreproj2 = transform3DPcl(_3dedges, R, t, K)
newEdges2 = generateEdgeImg(_2dreproj2, edges.shape[:])

edges_dt = cv2.distanceTransform(255 - edges,
                                 distanceType=cv.CV_DIST_L2,
                                 maskSize=cv.CV_DIST_MASK_PRECISE)
plt.figure()
plt.imshow(edges_dt, cmap=plt.get_cmap('gray'))
plt.title("distance transform")

plt.figure()
plt.imshow(edges, cmap=plt.get_cmap('gray'))
plt.title('orig edges')

plt.figure()
plt.imshow(newEdges, cmap=plt.get_cmap('gray'))
plt.title('new edges')

plt.figure()
plt.imshow(newEdges2, cmap=plt.get_cmap('gray'))
plt.title('new edges2')

new_edges_dt = cv2.distanceTransform(255 - newEdges,
                                     distanceType=cv.CV_DIST_L2,
                                     maskSize=cv.CV_DIST_MASK_PRECISE)
plt.figure()
plt.imshow(new_edges_dt, cmap=plt.get_cmap('gray'))
plt.title("distance transform new edges")

new_edges_dt2 = cv2.distanceTransform(255 - newEdges2,
                                      distanceType=cv.CV_DIST_L2,
                                      maskSize=cv.CV_DIST_MASK_PRECISE)
plt.figure()
plt.imshow(new_edges_dt2, cmap=plt.get_cmap('gray'))
plt.title("distance transform new edges")

#evaluate against three possibilities
sumOrig = 0.0
sumEdges1 = 0.0
sumSelf = 0.0
#now evaluate the distances
for xx in range(newEdges2.shape[1]):
    for yy in range(newEdges2.shape[0]):
        if (newEdges2[yy, xx] > 0):
            sumOrig += edges_dt[yy, xx]
            sumEdges1 += new_edges_dt[yy, xx]
            sumSelf += new_edges_dt2[yy, xx]
print((sumOrig, sumEdges1, sumSelf))
#plt.figure()
#plt.imshow(abs(edges-newEdges*255),cmap = plt.get_cmap('gray'))
#plt.title('diff edges')
