from config import Cfg
from .general_epipolar_constraint import *
from .camera_recovering import *
import time


def get_ransac_iterations(p_success=0.99,
                          outliers=0.5,
                          min_constraint=8):
    return int(
        np.log(1 - p_success) / np.log(1 - (1 - outliers) ** min_constraint)) + 1


class RANSAC_8PA:

    def __init__(self, cfg: Cfg):
        self.residual_threshold = cfg.params.residual_threshold
        self.probability_success = cfg.params.probability_success
        self.expected_inliers = cfg.params.expected_inliers
        self.solver = EightPointAlgorithmGeneralGeometry()
        self.max_trials = get_ransac_iterations(
            p_success=self.probability_success,
            outliers=1 - self.expected_inliers,
            min_constraint=8
        )
        self.num_samples = 0
        self.best_model = None
        self.best_evaluation = np.inf
        self.best_inliers = None
        self.best_inliers_num = 0
        self.counter_trials = 0
        self.time_evaluation = np.inf
        self.post_function_evaluation = None
        self.min_super_set = 8

    def estimate_essential_matrix(self, sample_bearings1, sample_bearings2, function):
        bearings = dict(
            x1=sample_bearings1,
            x2=sample_bearings2
        )

        return self.solver.get_e_from_cam_pose(function(**bearings))

    def run(self, bearings_1, bearings_2):
        assert bearings_1.shape == bearings_2.shape
        assert bearings_1.shape[0] is 3
        self.num_samples = bearings_1.shape[1]

        random_state = np.random.RandomState(1000)
        self.time_evaluation = 0
        aux_time = time.time()
        for self.counter_trials in range(self.max_trials):

            initial_inliers = random_state.choice(self.num_samples, self.min_super_set, replace=False)
            sample_bearings1 = bearings_1[:, initial_inliers]
            sample_bearings2 = bearings_2[:, initial_inliers]

            # * Estimation
            e_hat = self.solver.compute_essential_matrix(
                x1=sample_bearings1,
                x2=sample_bearings2,
            )

            # * Evaluation
            sample_residuals = self.solver.projected_error(
                e=e_hat,
                x1=bearings_1,
                x2=bearings_2
            )
            sample_evaluation = np.sum(sample_residuals ** 2)

            # * Selection
            sample_inliers = np.abs(sample_residuals) < self.residual_threshold
            sample_inliers_num = np.sum(sample_inliers)

            # * Loop Control
            lc_1 = sample_inliers_num > self.best_inliers_num
            lc_2 = sample_inliers_num == self.best_inliers_num
            lc_3 = sample_evaluation < self.best_evaluation
            if lc_1 or (lc_2 and lc_3):
                # + Update best performance
                self.best_model = e_hat.copy()
                self.best_inliers_num = sample_inliers_num.copy()
                self.best_evaluation = sample_evaluation.copy()
                self.best_inliers = sample_inliers.copy()

            if self.counter_trials >= self._dynamic_max_trials():
                break

        best_bearings_1 = bearings_1[:, self.best_inliers]
        best_bearings_2 = bearings_2[:, self.best_inliers]

        # * Estimating final model using only inliers
        self.best_model = self.estimate_essential_matrix(
            sample_bearings1=best_bearings_1,
            sample_bearings2=best_bearings_2,
            function=self.post_function_evaluation
            # ! predefined function used for post-evaluation
        )
        self.time_evaluation += time.time() - aux_time
        # * Final Evaluation
        sample_residuals = self.solver.projected_error(
            e=self.best_model,
            x1=best_bearings_1,
            x2=best_bearings_2
        )
        self.best_evaluation = np.sum(sample_residuals ** 2)

        # * Final Selection
        sample_inliers = sample_residuals < self.residual_threshold
        self.best_inliers_num = np.sum(sample_inliers)
        return self.best_model, self.best_inliers

    def get_cam_pose(self, bearings_1, bearings_2):
        self.run(
            bearings_1=bearings_1,
            bearings_2=bearings_2
        )
        cam_pose = self.solver.recover_pose_from_e(
            E=self.best_model,
            x1=bearings_1[:, self.best_inliers],
            x2=bearings_2[:, self.best_inliers]
        )
        return cam_pose

    def _dynamic_max_trials(self):
        if self.best_inliers_num == 0:
            return np.inf

        nom = 1 - self.probability_success
        if nom == 0:
            return np.inf

        inlier_ratio = self.best_inliers_num / float(self.num_samples)
        denom = 1 - inlier_ratio ** 8
        if denom == 0:
            return 1
        elif denom == 1:
            return np.inf

        nom = np.log(nom)
        denom = np.log(denom)
        if denom == 0:
            return 0

        return int(np.ceil(nom / denom))
