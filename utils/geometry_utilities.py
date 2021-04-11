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
