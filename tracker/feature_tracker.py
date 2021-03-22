from config import Cfg
from utils import get_dataset
from tracker.lkt_tracker import LKT_tracker
from tracker.shi_tomosi_extractor import Shi_Tomasi_Extractor
from tracker.frame import Frame
import numpy as np
import cv2


class FeatureTracker:
    def __init__(self, cfg: Cfg):
        self.cfg = cfg
        self.feature_extractor = Shi_Tomasi_Extractor(cfg)
        self.tracker = LKT_tracker(cfg)
        self.dataset = get_dataset(cfg)
        self.idx = cfg.initial_frame

    def track(self, return_dict=False):
        idx_curr = self.idx
        if idx_curr == self.dataset.number_frames:
            return None, None, None, False

        ret = True
        while True:
            frame_curr = Frame(**self.dataset.get_frame(idx_curr, return_dict=True))

            if self.idx == idx_curr:
                self.tracker.set_initial_frame(
                    initial_frame=frame_curr,
                    extractor=self.feature_extractor,
                )
                idx_curr += 1
                continue

            relative_pose = frame_curr.get_relative_pose(
                key_frame=self.tracker.initial_frame)
            camera_distance = np.linalg.norm(relative_pose[0:3, 3])

            tracked_img = self.tracker.track(frame=frame_curr)
            if self.cfg.show_tracked_features:
                print("Camera Distance       {}".format(camera_distance))
                print("Tracked features      {}".format(len(self.tracker.tracks)))
                print("KeyFrame/CurrFrame:   {}-{}".format(
                    self.tracker.initial_frame.idx, frame_curr.idx
                ))

                cv2.imshow("preview", tracked_img[:, :, ::-1])
                cv2.waitKey(1)

            if camera_distance > self.cfg.min_cam_distance:
                break
            idx_curr += 1
            if idx_curr == self.dataset.number_frames:
                return None, None, None, False

        relative_pose = frame_curr.get_relative_pose(key_frame=self.tracker.initial_frame)

        cam = self.dataset.cam
        matches = self.tracker.get_matches()
        try:
            bearings_kf = cam.pixel2euclidean_space(matches[0])
            bearings_frm = cam.pixel2euclidean_space(matches[1])
        except:
            bearings_kf = None
            bearings_frm = None
            print("Error projecting features to bearing vectors!!!")

        if self.cfg.special_tracking:
            self.idx += 1
        else:
            self.idx = self.tracker.frame_idx

        if return_dict:
            return dict(
                bearings_kf=bearings_kf,
                bearings_frm=bearings_frm,
                relative_pose=relative_pose,
                cfg=self.cfg
            ), ret
        return bearings_kf, bearings_frm, relative_pose, ret


if __name__ == '__main__':

    config_file = Cfg.FILE_CONFIG_MP3D_VO
    config_file = Cfg.FILE_CONFIG_TUM_VI

    cfg = Cfg.from_cfg_file(yaml_config=config_file)
    tracker = FeatureTracker(cfg)

    while True:
        bearings_kf, bearings_frm, cam_pose_gt, ret = tracker.track()
        if not ret:
            break
