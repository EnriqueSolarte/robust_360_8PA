import numpy as np
from geometry_utilities import *


def Normalization(nd, x):
    '''
    Normalization of coordinates (centroid to the origin and mean distance of sqrt(2 or 3).

    Input
    -----
    nd: number of dimensions, 3 here
    x: the data to be normalized (directions at different columns and points at rows)
    Output
    ------
    Tr: the transformation matrix (translation plus scaling)
    x: the transformed data
    '''

    x = np.asarray(x)
    m, s = np.mean(x, 0), np.std(x)
    if nd == 2:
        Tr = np.array([[s, 0, m[0]], [0, s, m[1]], [0, 0, 1]])
    else:
        Tr = np.array([[s, 0, 0, m[0]], [0, s, 0, m[1]], [0, 0, s, m[2]],
                       [0, 0, 0, 1]])

    Tr = np.linalg.inv(Tr)
    x = np.dot(Tr, np.concatenate((x.T, np.ones((1, x.shape[0])))))
    x = x[0:nd, :].T

    return Tr, x


def DLTcalib(nd, xyz, uvw):
    '''
    Camera calibration by DLT using known object points and their image points.

    Input
    -----
    nd: dimensions of the object space, 3 here.
    xyz: coordinates in the object 3D space.
    uv: coordinates in the image 2D space.

    The coordinates (x,y,z and u,v) are given as columns and the different points as rows.

    There must be at least 6 calibration points for the 3D DLT.

    Output
    ------
     L: array of 11 parameters of the calibration matrix.
     err: error of the DLT (mean residual of the DLT transformation in units of camera coordinates).
    '''
    if (nd != 3):
        raise ValueError('%dD DLT unsupported.' % (nd))

    # Converting all variables to numpy array
    xyz = np.asarray(xyz)
    uvw = np.asarray(uvw)

    n = xyz.shape[0]

    # Validating the parameters:
    if uvw.shape[0] != n:
        raise ValueError(
            'Object (%d points) and image (%d points) have different number of points.'
            % (n, uvw.shape[0]))

    if (xyz.shape[1] != 3):
        raise ValueError(
            'Incorrect number of coordinates (%d) for %dD DLT (it should be %d).'
            % (xyz.shape[1], nd, nd))

    if (n < 6):
        raise ValueError(
            '%dD DLT requires at least %d calibration points. Only %d points were entered.'
            % (nd, 2 * nd, n))

    # Normalize the data to improve the DLT quality (DLT is dependent of the system of coordinates).
    # This is relevant when there is a considerable perspective distortion.
    # Normalization: mean position at origin and mean distance equals to 1 at each direction.
    Txyz, xyzn = Normalization(nd, xyz)
    Tuvw, uvwn = Normalization(3, uvw)

    A = []

    for i in range(n):
        x, y, z = xyzn[i, 0], xyzn[i, 1], xyzn[i, 2]
        u, v, w = uvwn[i, 0], uvwn[i, 1], uvwn[i, 2]
        A.append(
            [x, y, z, 1, 0, 0, 0, 0, 0, 0, 0, 0, -u * x, -u * y, -u * z, -u])
        A.append(
            [0, 0, 0, 0, x, y, z, 1, 0, 0, 0, 0, -v * x, -v * y, -v * z, -v])
        A.append(
            [0, 0, 0, 0, 0, 0, 0, 0, x, y, z, 1, -w * x, -w * y, -w * z, -w])

        #     JUST FOR TESTING
        # bearing = uvwn[i, :]
        # matrix_vector = vector2skew_matrix(bearing)
        # p_3d_vector = np.ones((4, 1))
        # p_3d_vector[0:3, 0] = xyzn[i, :]
        # print("done")

    # Convert A to array
    A = np.asarray(A)

    # Find the 11 parameters:
    U, S, V = np.linalg.svd(A)

    # The parameters are in the last line of Vh and normalize them
    L = V[-1, :] / V[-1, -1]
    # print(L)
    # Camera projection matrix
    H = L.reshape(4, nd + 1)
    # print(H)

    # Denormalization
    # pinv: Moore-Penrose pseudo-inverse of a matrix, generalized inverse of a matrix using its SVD
    H = np.dot(np.dot(np.linalg.pinv(Tuvw), H), Txyz)
    # print(H)
    H = H / H[-1, -1]
    # print(H)
    L = H.flatten()
    # print(L)

    # Mean error of the DLT (mean residual of the DLT transformation in units of camera coordinates):
    uvw2 = np.dot(H, np.concatenate((xyz.T, np.ones((1, xyz.shape[0])))))
    uvw2 = uvw2 / uvw2[3, :]

    # Mean distance:
    err = np.sqrt(np.mean(np.sum((uvw2[0:3, :].T - uvw)**2, 1)))

    # from geometry_utilities import evaluate_error_in_transformation
    # err = evaluate_error_in_transformation(transform_gt=uvw,
    #                                        transform_est=uvw2)

    return L, err


def DLT(xyz, uvw):
    nd = 3
    P, err = DLTcalib(nd, xyz, uvw)
    print('Matrix')
    print(P)
    print('\nError')
    print(err)
    return P


if __name__ == "__main__":
    # Known 3D coordinates
    xyz = [[-875, 0, 9.755], [442, 0, 9.755], [1921, 0, 9.755],
           [2951, 0.5, 9.755], [-4132, 0.5, 23.618], [-876, 0, 23.618]]
    # Known pixel coordinates
    # uvw = [[76, 706, 1], [702, 706, 1], [1440, 706, 1], [1867, 706, 1], [264, 523, 1], [625, 523, 1]]
    uvw = [[76, 706, 1], [702, 706, 1], [1440, 706, 1], [1867, 706, 1],
           [264, 523, 1], [625, 523, 1]]

    DLT(xyz=xyz, uvw=uvw)
