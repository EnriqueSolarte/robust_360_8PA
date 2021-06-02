import numpy as np
import cv2

class UndistortedBase:
    """
    This is the Undistorter bases from which different distortion models will be defined
    """

    #! TODO Only ran-tan model has been implemented
    # region params property
    @property
    def params(self):
        return self.__dist_params

    @params.setter
    def params(self, value):
        self.__dist_params = value[0]
        assert value[1] in [
            "rad-tan"], "Distortion model {} does not exist".format(value[1])
        self.__dist_type = value[1]
        self.isThereUndistorter = True

    # endregion

    def __init__(self, parameters=None, dist_type=None):
        self.isThereUndistorter = False
        self.__dist_params = None
        self.params = [parameters, dist_type]

    def undistort(self, image):
        raise NotImplemented


class UndistorterRadTan(UndistortedBase):
    """
    Undistorter for Radial-tangencial distortions 
    """
    def __init__(self, K, coefs):
        """
        :param K: Camera intrinsc matrix (3, 3)
        :type K: numpy array
        :param coefs: k1, k2, k3, k4 coeficients parameters
        :type coefs: tuple
        """
        assert len(coefs) == 4
        params = dict(parameters=coefs,
                      dist_type='rad-tan')
        super(UndistorterRadTan, self).__init__(**params)
        assert K.shape == (3, 3)
        self.K = K

    def undistort(self, image):
        dst = cv2.undistort(image, self.K, self.params, None, None)
        return dst


if __name__ == "__main__":
    # * Undistorter parameters
    params = dict(K=np.eye(3),
                  coefs=[0.01, 0.02, 0.4, 0.5])

    undistorter = UndistorterRadTan(**params)
    print("done")
