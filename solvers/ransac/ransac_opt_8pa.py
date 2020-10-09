from solvers.ransac.ransac_8pa import *
from analysis.utilities.camera_recovering import *


class RANSAC_OPT_8PA(RANSAC_8PA):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.min_super_set = kwargs.get("min_set", 100)

    def run(self, bearings_1, bearings_2):
        assert bearings_1.shape == bearings_2.shape
        assert bearings_1.shape[0] is 3
        self.num_samples = bearings_1.shape[1]

        random_state = np.random.RandomState(1000)
        self.time_evaluation = 0
        aux_time = time.time()

        initial_inliers = random_state.choice(self.num_samples, self.min_super_set, replace=False)
        sample_bearings1 = bearings_1[:, initial_inliers]
        sample_bearings2 = bearings_2[:, initial_inliers]
        # sample_bearings1 = bearings_1
        # sample_bearings2 = bearings_2

        # ! Estimation
        e_hat = self.estimate_essential_matrix(
            sample_bearings1=sample_bearings1,
            sample_bearings2=sample_bearings2,
            function=get_e_by_8pa
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

        best_bearings_1 = bearings_1[:, self.best_inliers]
        best_bearings_2 = bearings_2[:, self.best_inliers]
        self.best_model = self.estimate_essential_matrix(
            sample_bearings1=best_bearings_1,
            sample_bearings2=best_bearings_2,
            function=get_cam_pose_by_opt_res_error_SK
        )

        self.time_evaluation += time.time() - aux_time
        aux_time = time.time()

        return self.best_model, self.best_inliers
