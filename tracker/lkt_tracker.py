import cv2
import numpy as np
from config import Cfg


class LKT_tracker:
    def __init__(self, cfg: Cfg):
        self.initial_frame = None
        self.tracked_frame = None
        self.prev_img = None
        self.curr_img = None
        self.visualization_image = None
        self.extractor = None
        self.tracks = []
        self.frame_idx = 0
        self.track_len = 0
        self.lk_params = dict(winSize=(cfg.block_size_for_tracking, cfg.block_size_for_tracking),
                              maxLevel=cfg.coarse_fine_levels,
                              criteria=(cv2.TERM_CRITERIA_EPS
                                        | cv2.TERM_CRITERIA_COUNT, cfg.counter_iterations, cfg.eps_tracking))

    def track(self, frame):
        self.tracked_frame = frame
        self.frame_idx = frame.idx
        visualization_img = np.array(frame.color_map).astype(np.uint8)
        self.curr_img = frame.grey_map

        if len(self.tracks) > 0:
            img0, img1 = self.prev_img, self.curr_img

            # TODO: Maybe we don't need to compute twice LKT
            p0 = np.float32([np.squeeze(tr[-1])
                             for tr in self.tracks]).reshape(-1, 1, 2)
            p1, _st, _err = cv2.calcOpticalFlowPyrLK(img0, img1, p0, None,
                                                     **self.lk_params)
            p0_hat, _st, _err = cv2.calcOpticalFlowPyrLK(
                img1, img0, p1, None, **self.lk_params)
            d = abs(p0 - p0_hat).reshape(-1, 2).max(-1)
            good = d < 1
            new_tracks = []

            # TODO: we can speed up here!!!
            # 1. Evaluating per tracked feature is wasted of time
            # 2. For visualization we can use a separated thread.
            for tr, (x, y), good_flag in zip(self.tracks, p1.reshape(-1, 2),
                                             good):
                if not good_flag:
                    continue
                tr.append((x, y))
                new_tracks.append(tr)

                cv2.circle(visualization_img, (x, y), 2, (0, 255, 0, 50), -1)
                cv2.drawMarker(visualization_img, (x, y), (0, 255, 0, 50), cv2.MARKER_SQUARE, 10)

            self.tracks = new_tracks
            cv2.polylines(visualization_img,
                          [np.int32(tr) for tr in self.tracks], False,
                          (0, 255, 0))

        self.prev_img = self.curr_img
        return visualization_img

    def set_initial_frame(self, initial_frame, extractor, mask=None):
        self.extractor = extractor
        self.initial_frame = initial_frame
        self.initial_frame.define_features(extractor, mask=mask)
        self.prev_img = self.initial_frame.grey_map
        self.tracks = []
        if self.initial_frame.key_points is not None:
            for key_point in self.initial_frame.key_points:
                self.tracks.append([(key_point.pt[0], key_point.pt[1])])

    def get_matches(self):
        matches = np.array([((tr[0]), (tr[-1]))
                            for tr in self.tracks]).reshape(
            (-1, 4)).astype(int)
        return matches[:, 0:2], matches[:, 2:4]
