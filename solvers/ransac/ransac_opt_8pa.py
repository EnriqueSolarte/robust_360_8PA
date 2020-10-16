from solvers.ransac.ransac_8pa import *
from analysis.utilities.camera_recovering import *


class RANSAC_OPT_8PA(RANSAC_8PA):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.min_super_set = kwargs.get("min_super_set", 100)
        self.relaxed_threshold = kwargs.get("relaxed_threshold", 1e-4)

    def run(self, bearings_1, bearings_2):
        assert bearings_1.shape == bearings_2.shape
        assert bearings_1.shape[0] is 3
        self.num_samples = bearings_1.shape[1]

        aux_threshold = self.relaxed_threshold
        random_state = np.random.RandomState(1000)
        self.time_evaluation = 0
        aux_time = time.time()
        for self.counter_trials in range(self.max_trials):
            initial_inliers = random_state.choice(self.num_samples, self.min_super_set, replace=False)
            sample_bearings1 = bearings_1[:, initial_inliers]
            sample_bearings2 = bearings_2[:, initial_inliers]
            # ! Estimation
            e_hat = self.estimate_essential_matrix(
                sample_bearings1=sample_bearings1,
                sample_bearings2=sample_bearings2,
                function=self.prior_function_evaluation
            )
            # * Evaluation
            sample_residuals = projected_error(
                e=e_hat,
                x1=bearings_1,
                x2=bearings_2
            )
            sample_evaluation = np.sum(sample_residuals ** 2)

            # * Selection
            sample_inliers = sample_residuals < aux_threshold
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

            if self.best_inliers_num >= self.expected_inliers * bearings_1.shape[1]:
                break
            elif self.counter_trials >= self._dynamic_max_trials():
                break
            else:
                aux_threshold += 0.1 * self.relaxed_threshold
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
        sample_residuals = projected_error(
            e=self.best_model,
            x1=best_bearings_1,
            x2=best_bearings_2
        )
        self.best_evaluation = np.sum(sample_residuals ** 2)

        # * Final Selection
        sample_inliers = sample_residuals < self.residual_threshold
        self.best_inliers_num = np.sum(sample_inliers)
        return self.best_model, self.best_inliers
