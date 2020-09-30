import numpy as np
from solvers.epipolar_constraint import *
import time


class RANSAC_8PA:

    def __init__(self, **kwargs):
        self.residual_threshold = kwargs.get("residual_threshold", 1e-3)
        self.probability_success = kwargs.get("probability_success", 0.99)
        self.expected_inliers = kwargs.get("expected_inliers", 0.5)
        self.solver = EightPointAlgorithmGeneralGeometry()
        self.max_trials = self.solver.get_ransac_iterations(
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

        # ! LOG
        self.log_data = dict()
        self.log_data["best_evaluation"] = []
        self.log_data["best_inliers_num"] = []

    def run(self, bearings_1, bearings_2):
        assert bearings_1.shape == bearings_2.shape
        assert bearings_1.shape[0] is 3
        self.num_samples = bearings_1.shape[1]

        random_state = np.random.RandomState(1000)
        self.time_evaluation = 0
        aux_time = time.time()
        for self.counter_trials in range(self.max_trials):
            # self.counter_trials += 1
            initial_inliers = random_state.choice(self.num_samples, 8, replace=False)
            sample_bearings1 = bearings_1[:, initial_inliers]
            sample_bearings2 = bearings_2[:, initial_inliers]
            # ! Estimation
            e_hat = self.solver.compute_essential_matrix(
                x1=sample_bearings1,
                x2=sample_bearings2,
            )
            # ! Evaluation
            sample_residuals = projected_error(
                e=e_hat,
                x1=bearings_1,
                x2=bearings_2
            )
            sample_evaluation = np.sum(sample_residuals ** 2)

            # ! Selection
            sample_inliers = sample_residuals < self.residual_threshold
            sample_inliers_num = np.sum(sample_inliers)

            # ! Loop Control
            lc_1 = sample_inliers_num > self.best_inliers_num
            lc_2 = sample_inliers_num == self.best_inliers_num
            lc_3 = sample_evaluation < self.best_evaluation
            if lc_1 or (lc_2 and lc_3):
                # ! Update best performance
                self.best_model = e_hat.copy()
                self.best_inliers_num = sample_inliers_num.copy()
                self.best_evaluation = sample_evaluation.copy()
                self.best_inliers = sample_inliers.copy()

            if self.counter_trials >= self._dynamic_max_trials():
                break
            self.time_evaluation += time.time() - aux_time
            aux_time = time.time()

            # ! saving log
            self.log_data["best_evaluation"].append(self.best_evaluation.copy())
            self.log_data["best_inliers_num"].append(self.best_inliers_num.copy())

        best_bearings_1 = bearings_1[:, self.best_inliers]
        best_bearings_2 = bearings_2[:, self.best_inliers]
        self.best_model = self.solver.compute_essential_matrix(
            x1=best_bearings_1,
            x2=best_bearings_2,
        )

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