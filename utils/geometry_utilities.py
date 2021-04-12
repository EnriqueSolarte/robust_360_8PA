import numpy as np
import math


def extend_array_to_homogeneous(array):
    """
    Returns the homogeneous form of a vector by attaching
    a unit vector as additional dimensions
    Parameters
    ----------
    array of (3, n) or (2, n)
    Returns (4, n) or (3, n)
    -------
    """
    try:
        assert array.shape[0] in (2, 3, 4)
        dim, samples = array.shape
        return np.vstack((array, np.ones((1, samples))))

    except:
        assert array.shape[1] in (2, 3, 4)
        array = array.T
        dim, samples = array.shape
        return np.vstack((array, np.ones((1, samples)))).T


def eulerAnglesToRotationMatrix(angles):
    theta = np.zeros((3))

    if angles.__class__.__name__ == 'dict':
        theta[0] = angles['x']
        theta[1] = angles['y']
        theta[2] = angles['z']
    else:
        theta[0] = angles[0]
        theta[1] = angles[1]
        theta[2] = angles[2]

    R_x = np.array([[1, 0, 0], [0, math.cos(theta[0]), -math.sin(theta[0])],
                    [0, math.sin(theta[0]),
                     math.cos(theta[0])]])

    R_y = np.array([[math.cos(theta[1]), 0,
                     math.sin(theta[1])], [0, 1, 0],
                    [-math.sin(theta[1]), 0,
                     math.cos(theta[1])]])

    R_z = np.array([[math.cos(theta[2]), -math.sin(theta[2]), 0],
                    [math.sin(theta[2]),
                     math.cos(theta[2]), 0], [0, 0, 1]])

    R = np.dot(R_z, np.dot(R_y, R_x))
    T = np.eye(4)
    T[0:3, 0:3] = R
    return T


def isRotationMatrix(R):
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype=R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-6


def extend_vector_to_homogeneous_transf(vector):
    """
    Creates a homogeneous transformation (4, 4) given a vector R3
    :param vector: vector R3 (3, 1) or (4, 1)
    :return: Homogeneous transformation (4, 4)
    """
    T = np.eye(4)
    if vector.__class__.__name__ == 'dict':
        T[0, 3] = vector['x']
        T[1, 3] = vector['y']
        T[2, 3] = vector['z']
    elif type(vector) == np.array:
        T[0:3, 3] = vector[0:3, 0]
    else:
        T[0:3, 3] = vector[0:3]
    return T


def get_homogeneous_transform_from_vectors(r_vector, t_vector):
    """
    Returns a homogeneous transformation (4x4) given rot vector (Euler angles)
    and translation vector. (Euler angles in degrees)
    Parameters
    ----------
    r_vector
    t_vector

    Returns
    -------
    4x4 homogeneous transformation
    """
    r_vector = np.array(r_vector)
    if r_vector.shape == (3, 3):
        if isRotationMatrix(r_vector):
            r_gt = r_vector
        else:
            assert EOFError
        t = np.eye(4)
        t[0:3, 0:3] = r_gt
        t[0:3, 3] = t_vector
        return t
    else:
        t_gt = extend_vector_to_homogeneous_transf(t_vector)

        r_gt = eulerAnglesToRotationMatrix({
            'x': np.radians(r_vector[0]),
            'y': np.radians(r_vector[1]),
            'z': np.radians(r_vector[2])
        })

        return np.matmul(t_gt, r_gt)


def spherical_normalization(array):
    assert array.shape[0] in (3, 4)
    if array.shape.__len__() < 2:
        array = array.reshape(-1, 1)
    norm = np.linalg.norm(array[0:3, :], axis=0)
    return array[0:3, :] / norm


def vector2skew_matrix(vector):
    """
    Converts a vector [3,] into a matrix [3, 3] for cross product operations. v x v' = [v]v' where [v] is a skew representation of v
    :param vector: [3,]
    :return: skew matrix [3, 3]
    """
    assert len(vector.shape) < 2

    skew_matrix = np.zeros((3, 3))
    skew_matrix[1, 0] = vector[2]
    skew_matrix[2, 0] = -vector[1]
    skew_matrix[0, 1] = -vector[2]
    skew_matrix[2, 1] = vector[0]
    skew_matrix[0, 2] = vector[1]
    skew_matrix[1, 2] = -vector[0]

    return skew_matrix.copy()


def skew_matrix2vector(matrix):
    assert matrix.shape == (3, 3)

    vector = np.zeros((3, 1))

    vector[0] = matrix[2, 1]
    vector[1] = -matrix[2, 0]
    vector[2] = matrix[1, 0]

    return vector


def evaluate_error_in_transformation(transform_est,
                                     transform_gt):
    """
    Return the angular error in rotation and translation as the drift angle between two vector
    for both rotation and translation
    Ref:
    Fathian, et.al. (RAL 2018). QuEst: A Quaternion-Based Approach for Camera.
    Huynh, D. Q. (2009). Metrics for 3D Rotations: Comparison and Analysis.
    """
    assert transform_est.shape == transform_gt.shape == (4, 4)
    # ! Error in rotation

    # q_gt = standard_quaternion(Quaternion(matrix=transform_gt[0:3, 0:3]))
    # q_st = standard_quaternion(Quaternion(matrix=transform_est[0:3, 0:3]))
    # q_gt = standard_quaternion(Quaternion(matrix=eulerAnglesToRotationMatrix((0, 0, 0))[0:3, 0:3]))
    # q_st = standard_quaternion(Quaternion(matrix=eulerAnglesToRotationMatrix((0, np.radians(90), 0))[0:3, 0:3]))
    # rot_err = (q_gt.inverse * q_st).radians / np.pi
    error = 0.5 * (np.trace(transform_gt[0:3, 0:3].T.dot(
        transform_est[0:3, 0:3])) - 1)
    rot_err = np.arccos(np.clip(error, -1, 1)) / np.pi

    # ! Error in translation
    trans_err = angle_between_vectors(transform_gt[0:3, 3],
                                      transform_est[0:3, 3]) / np.pi
  
    return np.abs(rot_err), np.abs(trans_err)


def angle_between_vectors(vect_ref, vect):
    """
    This function returns the angle between two vectors
    """

    c = np.dot(vect_ref.T, vect) / (np.linalg.norm(vect_ref) * np.linalg.norm(vect))
    angle = np.arccos(np.clip(c, -1, 1))

    return angle


def evaluate_error_in_essential_matrix(e_ref, e_hat):
    assert e_ref.shape == e_hat.shape
    assert e_ref.shape[0] == e_ref.shape[1] == 3

    e_ref = e_ref / np.linalg.norm(e_ref)
    e_hat = e_hat / np.linalg.norm(e_hat)

    return np.min(
        (np.linalg.norm(e_ref - e_hat), np.linalg.norm(e_ref + e_hat)))
