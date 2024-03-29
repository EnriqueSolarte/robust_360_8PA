from config import Cfg
from utils.data_utilities import get_dataset, sampling_idxs
from utils.geometry_utilities import get_homogeneous_transform_from_vectors, extend_array_to_homogeneous
from utils.pcl_utilities import add_noise_to_pcl
from utils.pcl_utilities import add_outliers_to_pcl
import numpy as np
import cv2

from utils.vispy_utilities import plot_color_plc, plot_pcl_list


class BearingsSampler:
    def __init__(self, cfg: Cfg):
        self.cfg = cfg
        assert cfg.params.dataset_name == "MP3D_VO", "Wrong datastet. Only MP3D-VO is suitable for sampling a PCL"
        self.dataset = get_dataset(cfg)
        self.idx = cfg.params.initial_frame

    def get_bearings(self, return_dict=False):
        idx_curr = self.idx

        if idx_curr < self.dataset.number_frames:
            ret = True
            pcl, color, pose, timestamp, idx = self.dataset.get_pcl(idx_curr)

            # ! Create a random camera pose
            linear_motion = self.cfg.params.range_liner_motion
            angular_motion = self.cfg.params.range_angular_motion
            cam_a2b = get_homogeneous_transform_from_vectors(
                t_vector=(np.random.uniform(linear_motion[0], linear_motion[1]),
                          np.random.uniform(linear_motion[0], linear_motion[1]),
                          np.random.uniform(linear_motion[0], linear_motion[1])),
                r_vector=(np.random.uniform(angular_motion[0], angular_motion[1]),
                          np.random.uniform(angular_motion[0], angular_motion[1]),
                          np.random.uniform(angular_motion[0], angular_motion[1])))

            # ! Sampling PCL
            samples = sampling_idxs(length=pcl.shape[1], max_size=self.cfg.params.max_number_features)      
            
            pcl_a = pcl[:, samples]

            pcl_b = np.linalg.inv(cam_a2b)[0:3, :] @ extend_array_to_homogeneous(pcl_a)

            pcl_b = add_noise_to_pcl(pcl_b, param=self.cfg.params.vMF_kappa)

            pcl_b = add_outliers_to_pcl(pcl_b, number_inliers=(1-self.cfg.params.outliers_ratio) * pcl_b.shape[1])

            bearings_kf = self.dataset.cam.sphere_normalization(pcl_a)
            bearings_frm = self.dataset.cam.sphere_normalization(pcl_b)

            self.idx += self.cfg.params.skipped_frames
        else:
            ret = False

        self.cfg.tracked_or_sampled = self.cfg.FROM_SAMPLED_BEARINGS
        if return_dict:

            if not ret:
                return None, ret

            return dict(
                bearings_kf=bearings_kf,
                bearings_frm=bearings_frm,
                relative_pose=cam_a2b,
                idx_kf=idx_curr,
                idx_frm=idx_curr+1,
                cfg=self.cfg
            ), ret

        if not ret:
            return None, None, None,  ret

        return bearings_kf, bearings_frm, cam_a2b, ret


if __name__ == '__main__':

    config_file = Cfg.FILE_CONFIG_MP3D_VO

    cfg = Cfg.from_cfg_file(yaml_config=config_file)
    tracker = BearingsSampler(cfg)

    while True:
        bearings_kf, bearings_frm, cam_pose_gt, ret = tracker.get_bearings()
        print("Ready!!! ")
        if not ret:
            break
