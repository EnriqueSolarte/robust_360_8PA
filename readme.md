# Robust 360-8PA

This is the implementation of our ICRA 2021 ["
Robust 360-8PA": Redesigning The Normalized 8-point Algorithm
for 360-FoV Images](https://arxiv.org/abs/2104.10900)" ([website project](https://enriquesolarte.github.io/robust_360_8pa/)).


### Introduction
[![image](https://github.com/EnriqueSolarte/EnriqueSolarte.github.io/blob/main/robust_360_8pa/assets/play_video_12_min.png)](https://drive.google.com/file/d/1BazLvaZllLIf-QY1xb2tbQaJYhtXy2_R/preview)

For a quick introduction (3 min), please [here](https://drive.google.com/file/d/1qSuaWtE9uO62hN8XR8Gxa4KDCfVfeDzt/preview)


---
### News
**June 3, 2021** - Code realese 

***Coming Soon***: *Dataset MP3D-VO realese*

### Description

This **REPO** is our own implemantation, in python, for a camera pose estimation using the eight-point algorithm [^1], the non-linear optmization over residual errors (Gold Standar Method [GSM]) [^2], and our method named **Robust 360-8PA**.

Using this implementation, you can:
* Track features from 360-FoV (our MP3D-VO dataset) and Fish-eye (TUM-VI dataset [^3]) images using LKT-tracker. 
* Sample 3D points from GT 360-depth maps (only for our MP3D-VO dataset), adding noise vMF, and outliers.
* Evaluate camera pose, with and without RANSAC, using 8-PA[^1], GSM[^2], and our **Robust 360-8PA**.


[^1]: [Longuet-Higgins, H. C. (1981). A computer algorithm for reconstructing a scene from two projections. Nature, 293(5828), 133-135.](https://www.nature.com/articles/293133a0)

[^2]: [A. Pagani and D. Stricker, "Structure from Motion using full spherical panoramic cameras," 2011 IEEE International Conference on Computer Vision Workshops (ICCV Workshops), 2011, pp. 375-382, doi: 10.1109/ICCVW.2011.6130266.](10.1109/ICCVW.2011.6130266)

[^3]: [Schubert, D., Goll, T., Demmel, N., Usenko, V., Stückler, J., & Cremers, D. (2018, October). The TUM VI benchmark for evaluating visual-inertial odometry. In 2018 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS) (pp. 1680-1687). IEEE.](https://arxiv.org/abs/1804.06120)