import cv2
from frame import Frame
import numpy as np


class LKTracker:
    def __init__(self, skip_frames=5, track_len=10):
        self.initial_frame = None
        self.tracked_frame = None
        self.prev_img = None
        self.curr_img = None
        self.extractor = None
        self.tracks = []
        self.frame_idx = 0
        self.track_len = 10
        self.detect_interval = skip_frames
        self.lk_params = dict(
            winSize=(20, 20),
            maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS
                      | cv2.TERM_CRITERIA_COUNT, 10, 0.001))

    def track(self, frame):
        self.tracked_frame = frame
        self.frame_idx = frame.idx
        visualization_img = frame.color_map
        self.curr_img = frame.grey_map

        if len(self.tracks) > 0:
            img0, img1 = self.prev_img, self.curr_img
            p0 = np.float32(
                [np.squeeze(tr[-1]) for tr in self.tracks]).reshape(-1, 1, 2)
            p1, _st, _err = cv2.calcOpticalFlowPyrLK(img0, img1, p0, None,
                                                     **self.lk_params)
            p0_hat, _st, _err = cv2.calcOpticalFlowPyrLK(
                img1, img0, p1, None, **self.lk_params)
            d = abs(p0 - p0_hat).reshape(-1, 2).max(-1)
            good = d < 50
            new_tracks = []
            for tr, (x, y), good_flag in zip(self.tracks, p1.reshape(-1, 2),
                                             good):
                if not good_flag:
                    continue
                tr.append((x, y))
                new_tracks.append(tr)
                # if len(tr) > self.track_len:
                #     del tr[0]
                cv2.circle(visualization_img, (x, y), 2, (0, 255, 0), -1)
            self.tracks = new_tracks
            cv2.polylines(visualization_img,
                          [np.int32(tr) for tr in self.tracks], False,
                          (0, 255, 0))

        # if self.frame_idx % self.detect_interval == 0:
        #     mask = np.zeros_like(self.curr_img)
        #     mask[:] = 255
        #     for x, y in [np.int32(tr[-1]) for tr in self.tracks]:
        #         cv2.circle(mask, (x, y), 5, 0, -1)
        #
        #     p, d = self.extractor.get_features_descriptors(self.curr_img, mask)
        #     if p is not None:
        #         for orb in p:
        #             self.tracks.append([(orb.pt[0], orb.pt[1])])

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
        matches = np.array(
            [((tr[0]), (tr[-1])) for tr in self.tracks]).reshape(
                (-1, 4)).astype(int)
        return matches[:, 0:2], matches[:, 2:4]
