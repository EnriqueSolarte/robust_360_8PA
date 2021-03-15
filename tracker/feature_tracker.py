from config import Cfg
import utils
from .lkt_tracker import LKT_tracker
from .shi_tomosi_extractor import Shi_Tomasi_Extractor
from .frame import Frame

class FeatureTracker:
    def __init__(self, cfg:Cfg):
        self.cfg
        self.feature_extractor = Shi_Tomasi_Extractor(cfg)
        self.tracker = LKT_tracker(cfg)
        self.dataset = utils.get_dataset(cfg)
        self.idx_frame = cfg.initial_frame
    

    def track(self):
        
        curr_frame = Frame(self.dataset.get_frame(self.idx_frame, return_dict=True))
        
        self.tracker.set_initial_frame(
            initial_frame=curr_frame,
            extractor=self.feature_extractor,
            mask=utils.get_mask(self.cfg)
        )
        print("done")