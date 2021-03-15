from config import Cfg
from tracker import FeatureTracker


def from_tracked_features(cfg: Cfg):
    tracker = FeatureTracker(cfg)

    while True:
        data_bearings, ret =  tracker.track()
        if not ret:
            break
        print("done")
