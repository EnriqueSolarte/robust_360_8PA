# Robust 360-8PA

This is the implementation of our ICRA 2021 ["
Robust 360-8PA": Redesigning The Normalized 8-point Algorithm
for 360-FoV Images](https://arxiv.org/abs/2104.10900)" ([website project](https://enriquesolarte.github.io/robust_360_8pa/)).


### Introduction
[![image](https://github.com/EnriqueSolarte/EnriqueSolarte.github.io/blob/main/robust_360_8pa/assets/play_video_12_min.png)](https://drive.google.com/file/d/1BazLvaZllLIf-QY1xb2tbQaJYhtXy2_R/preview)

For a quick introduction (3 min), please [here](https://drive.google.com/file/d/1qSuaWtE9uO62hN8XR8Gxa4KDCfVfeDzt/preview)


---
### News
**June 3, 2021** - Code release 

***Coming Soon***: *Dataset MP3D-VO release*

---

### Description

This **REPO** is our own implemantation, in python, for a camera pose estimation using the eight-point algorithm [1], the non-linear optmization over residual errors (Gold Standar Method [GSM]) [2], and our method named **Robust 360-8PA**.

Using this implementation, you can:

* Track key-features, using LKT-tracker, from 360-FoV and Fish-eye images (from our MP3D-VO and TUM-VI[3] datasets, respectively). 
* Sample 3D points from GT 360-depth maps (only for our MP3D-VO dataset), adding noise vMF, and outliers.
* Evaluate camera pose, with and without RANSAC, using 8-PA[1], GSM[2], and our **Robust 360-8PA**.

Futher capabilities, analysis and resourses are released in the branch [```dev```](https://github.com/EnriqueSolarte/robust_360_8PA/tree/dev).

---
### Requirements
* python                    3.7.7
* vispy                     0.5.3
* numpy                     1.18.5 
* opencv-python             3.4.3.18
* pandas                    1.0.5 
* levmar                    0.2.3
---
### Settings

For convience, we implement a  ***< Class config >*** to load the used settings in this repo, from a yaml file. ```e.g .config/config_TUM_VI.yaml```. You can use the following lines for loading this configuracion. 

```py
from config import Cfg

 config_file = Cfg.FILE_CONFIG_MP3D_VO # PATH to yaml file
 cfg = Cfg.from_cfg_file(yaml_config=config_file)
```

the ```cfg``` instance is used to set all of the classes and methods in this implementation. e.g., 

```py

# from test/test_tracking_features.py
config_file = Cfg.FILE_CONFIG_TUM_VI    
cfg = Cfg.from_cfg_file(yaml_config=config_file)
tracker = FeatureTracker(cfg)

# from test/test_saving_sampled_bearings.py
config_file = Cfg.FILE_CONFIG_MP3D_VO
cfg = Cfg.from_cfg_file(yaml_config=config_file)
sampler = BearingsSampler(cfg)

# test/test_eval_methods.py
config_file = Cfg.FILE_CONFIG_MP3D_VO
cfg = Cfg.from_cfg_file(yaml_config=config_file)
eval_solvers(cfg)

```
***NOTE***: In general ```cfg``` is created at the begining of every script in this implementation, e.g. in  ```plots/plot_cam_pose_errors.py```

```py
if __name__ == '__main__':
    config_file = Cfg.FILE_CONFIG_MP3D_VO
    cfg = Cfg.from_cfg_file(yaml_config=config_file)
    plot_sampling_evaluations(cfg)
```

##### ENV variables and source this implementation


There are three main ENV variables that have to be modified in ```env``` file. 

```sh
DIR_DATASETS=/HD/datasets  #path to root dir of datasets
MP3D_VO_DATASET=${DIR_DATASETS}/ICRA2021  # path to MP3D-VO dataset
TUM_VI_DATASET=${DIR_DATASETS}/TUM_VI       # path to TUM-VI dataset
```
After this. You should source the ```setup.sh``` to export these variables and set the current implementation into your PYTHONPATH. This can be added to your .bashrc file if your are using LINUX.

```sh
# from root of this REPO
source setup.bash
```
---
## RUN FILES

To run this implementation, we present several run test files in ```/test```:
* ```/test/test_tracking_features.py```
* ```/test/test_saving_tracked_features.py```
* ```/test/test_saving_sampled_bearings.py```
* ```/test/test_read_saved_bearings.py```
* ```/test/test_eval_methods.py```
* ```/test/test_eval_ransac_methods.py  ```

In order to run one of these files, you just need to execute them as normal python script. e.g.
 ```
 python test/test_eval_ransac_methods.py
 ```

output
```
Number of evaluated bearings: 400
Rot-e:6.135843e-02      Tran-e:1.302335e-01     ransac_8pa
Rot-e:5.519852e-02      Tran-e:8.634036e-02     ransac_opt_SK
Rot-e:6.127112e-02      Tran-e:9.963809e-02     ransac_GSM
Rot-e:3.659004e-02      Tran-e:8.599553e-02     ransac_GSM_const_wRT
Rot-e:2.893354e-02      Tran-e:5.370007e-02     ransac_GSM_const_wSK
```

To plot the MAE evaluations (median absolute errors), you can run:

```
python plots/plot_cam_pose_errors.py
```

[![image](https://github.com/EnriqueSolarte/EnriqueSolarte.github.io/blob/main/robust_360_8pa/assets/demo_v3_2.gif)](https://enriquesolarte.github.io/robust_360_8pa/)


## Acknowledgement
- Credit of this repo is shared with [Chin-Hsuan Wu](https://chinhsuanwu.github.io/).

## Citation
Please cite our paper for any purpose of usage.
```
 @misc{solarte2021robust,
                    title={Robust 360-8PA: Redesigning The Normalized 8-point Algorithm for 360-FoV Images}, 
                    author={Bolivar Solarte and Chin-Hsuan Wu and Kuan-Wei Lu and Min Sun and Wei-Chen Chiu and Yi-Hsuan Tsai},
                    year={2021},
                    eprint={2104.10900},
                    archivePrefix={arXiv},
                    primaryClass={cs.CV}
}
```
---
### References
[1]: [Longuet-Higgins, H. C. (1981). A computer algorithm for reconstructing a scene from two projections. Nature, 293(5828), 133-135.](https://www.nature.com/articles/293133a0)

[2]: [A. Pagani and D. Stricker, "Structure from Motion using full spherical panoramic cameras," 2011 IEEE International Conference on Computer Vision Workshops (ICCV Workshops), 2011, pp. 375-382, doi: 10.1109/ICCVW.2011.6130266.](10.1109/ICCVW.2011.6130266)

[3]: [Schubert, D., Goll, T., Demmel, N., Usenko, V., Stückler, J., & Cremers, D. (2018, October). The TUM VI benchmark for evaluating visual-inertial odometry. In 2018 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS) (pp. 1680-1687). IEEE.](https://arxiv.org/abs/1804.06120)
