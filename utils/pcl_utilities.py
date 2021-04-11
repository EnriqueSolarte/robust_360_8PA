
import numpy as np
from scipy.stats import vonmises


def add_outliers_to_pcl(pcl, number_inliers=1):
    """
    It randomly select vector into the pcl array and redefines its values
    """
    assert pcl.shape[0] in (3, 4)
    assert number_inliers <= pcl.shape[1]
    outliers_src = np.random.randint(0, pcl.shape[1], pcl.shape[1] - int(number_inliers))
    outliers_des = np.random.randint(0, pcl.shape[1], pcl.shape[1] - int(number_inliers))
    outliers_sign = np.random.choice((-1, 1), size=pcl.shape[1] - int(number_inliers))

    pcl[:, outliers_src] = outliers_sign * pcl[:, outliers_des]

    return pcl


def add_noise_to_pcl(pcl, param, noisy_type="vmf"):
    """
    Return a noisy pcl based on the given pcl as parameter.
    noisy_type: ("vmf", "gaussian")
    """
    assert pcl.shape[0] in (3, 4)
    assert noisy_type in ("vmf", "gaussian")

    if noisy_type == "gaussian":
        pcl += param * np.random.random(pcl.shape)
    else:
        pcl += vonmises.rvs(param, size=pcl.shape)

    return pcl
