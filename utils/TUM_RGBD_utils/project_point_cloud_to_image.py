# -*- coding: utf-8 -*-
"""
Created on Tue Aug 16 13:25:17 2016

@author: fschenk
"""

import cv2
import cv
import numpy as np
import matplotlib.pyplot as plt

test = []

res = cv.Load(
    '/datawl/CSI/CSI_SVN/Kiras CSISmartCam3D/CSISmartScan3D_Daten/Software/InfiniTAM-build/my.xml'
)
pcl = cv.Load(
    '/datawl/CSI/CSI_SVN/Kiras CSISmartCam3D/CSISmartScan3D_Daten/Software/InfiniTAM-build/pcl.xml'
)
#res = cv2.imread('/datawl/CSI/CSI_SVN/Kiras CSISmartCam3D/CSISmartScan3D_Daten/Daten_Testszenen/TUM/raycast_image.exr');
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
res = pcl
d = np.zeros([480, 640])
for xx in range(640):
    for yy in range(480):
        _2d = K.dot(res[yy, xx][0:3])
        x = _2d[0] / _2d[2]
        y = _2d[1] / _2d[2]
        if (x >= 0 and x < 640 and y >= 0 and y < 480):
            d[y, x] = res[yy, xx][2]
d2 = d * 1.0 / 0.005
plt.imshow(d2)
stop
