"""
Transforms Module - Contains code for to learn about rotations
and eventually homogenous transforms.

Empty outline derived from code written by John Morrell, former TA.
"""

from matplotlib.pylab import norm
import numpy as np
from numpy import sin, cos, sqrt
from numpy.typing import NDArray
from utility_fp import clean_rotation_matrix


## 2D Rotations
def rot2(theta: float) -> NDArray:
    """
    R = rot2(th)

    :param float theta: angle of rotation (rad)
    :return R: 2x2 numpy array representing rotation in 2D by theta
    """

    ## TODO - Fill this out
    R = np.array([[cos(theta), -sin(theta)],
                   [sin(theta), cos(theta)]])
    return clean_rotation_matrix(R)


## 3D Transformations
def rotx(theta: float) -> NDArray:
    """
    R = rotx(theta)

    :param float theta: angle of rotation (rad)
    :return R: 3x3 numpy array representing rotation about x-axis by amount theta
    """
    ## TODO - Fill this out
    R = np.array([[1, 0, 0],
                   [0, cos(theta), -sin(theta)],
                   [0, sin(theta), cos(theta)]])
    return clean_rotation_matrix(R)

def sp_rotx(theta):
    return sp.Matrix([[1, 0, 0],
                      [0, sp.cos(theta), -sp.sin(theta)],
                      [0, sp.sin(theta), sp.cos(theta)]])


def roty(theta: float) -> NDArray:
    """
    R = roty(theta)

    :param float theta: angle of rotation (rad)
    :return R: 3x3 numpy array representing rotation about y-axis by amount theta
    """
    ## TODO - Fill this out
    R = np.array([[cos(theta), 0, sin(theta)],
                     [0, 1, 0],
                     [-sin(theta), 0, cos(theta)]])
    return clean_rotation_matrix(R)

def sp_roty(theta):
    return sp.Matrix([[sp.cos(theta), 0, sp.sin(theta)],
                      [0, 1, 0],
                      [-sp.sin(theta), 0, sp.cos(theta)]])


def rotz(theta: float) -> NDArray:
    """
    R = rotz(theta)

    :param float theta: angle of rotation (rad)
    :return R: 3x3 numpy array representing rotation about z-axis by amount theta
    """
    ## TODO - Fill this out
    R = np.array([[cos(theta), -sin(theta), 0],
                   [sin(theta), cos(theta), 0],
                   [0, 0, 1]])

    return clean_rotation_matrix(R)

def sp_rotz(theta):
    return sp.Matrix([[sp.cos(theta), -sp.sin(theta), 0],
                      [sp.sin(theta), sp.cos(theta), 0],
                      [0, 0, 1]])


# inverse of rotation matrix
def rot_inv(R: NDArray) -> NDArray:
    '''
    R_inv = rot_inv(R)

    :param NDArray R: 2x2 or 3x3 numpy array representing a proper rotation matrix
    :return R_inv: 2x2 or 3x3 inverse of the input rotation matrix
    '''
    ## TODO - Fill this out
    R_inv = np.linalg.inv(R)
    return R_inv


def se3(R: NDArray=np.eye(3), p: NDArray=np.zeros(3)) -> NDArray:
    """
    T = se3(R, p)

    Creates a 4x4 homogeneous transformation matrix "T" from a 3x3 rotation matrix
    and a position vector.

    :param NDArray R: 3x3 numpy array representing orientation, defaults to identity.
    :param NDArray p: numpy array representing position, defaults to [0, 0, 0].
    :return T: 4x4 numpy array representing the homogeneous transform.
    """
    # TODO - fill out "T"
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = p

    return T

import sympy as sp

def sp_se3(R=sp.eye(3), p=sp.zeros(3, 1)):
    """
    T = se3(R, p)

    Creates a 4x4 homogeneous transformation matrix "T" from a 3x3 rotation matrix
    and a position vector, compatible with SymPy.

    :param R: 3x3 SymPy Matrix representing orientation, defaults to identity.
    :param p: 3x1 SymPy Matrix representing position, defaults to [0, 0, 0].
    :return T: 4x4 SymPy Matrix representing the homogeneous transform.
    """
    T = sp.eye(4)  # Create a 4x4 identity matrix
    T[:3, :3] = R  # Set the top-left 3x3 block to the rotation matrix
    T[:3, 3] = p   # Set the top-right 3x1 block to the position vector
    return T

def inv(T: NDArray) -> NDArray:
    """
    T_inv = inv(T)

    Returns the inverse transform to T.

    :param NDArray T: 4x4 homogeneous transformation matrix
    :return T_inv: 4x4 numpy array that is the inverse to T so that T @ T_inv = I
    """

    #TODO - fill this out
    R = T[:3, :3]
    p = T[:3, 3]
    R_inv = rot_inv(R)
    p_inv = -R_inv @ p
    T_inv = se3(R_inv, p_inv)

    return T_inv

def R2rpy(R: NDArray) -> NDArray:
    """
    rpy = R2rpy(R)

    Returns the roll-pitch-yaw representation of the SO3 rotation matrix.

    :param NDArray R: 3x3 Numpy array for any rotation.
    :return rpy: Numpy array, containing [roll, pitch, yaw] coordinates (in radians).
    """

    # follow formula in book, use functions like "np.atan2"
    # for the arctangent and "**2" for squared terms.
    # TODO - fill out this equation for rpy

    roll = np.atan2(R[1, 0], R[0, 0])
    pitch = np.atan2(-R[2, 0], np.sqrt(R[2, 1]**2 + R[2, 2]**2))
    yaw = np.atan2(R[2, 1], R[2, 2])

    return np.array([roll, pitch, yaw])


def R2axis(R: NDArray) -> NDArray:
    """
    axis_angle = R2axis(R)

    Returns an axis angle representation of a SO(3) rotation matrix.

    :param NDArray R: 3x3 rotation matrix.
    :return axis_angle: numpy array containing the axis angle representation
        in the form: [angle, rx, ry, rz]
    """

    # see equation (2.27) and (2.28) on pg. 54, using functions like "np.acos," "np.sin," etc.
    ang = np.arccos((np.trace(R) - 1) / 2)
    axis_angle = np.array([ang,
        (R[2, 1] - R[1, 2])/(2*np.sin(ang)),
        (R[0, 2] - R[2, 0])/(2*np.sin(ang)),
        (R[1, 0] - R[0, 1])/(2*np.sin(ang))])

    return axis_angle


def axis2R(angle: float, axis: NDArray) -> NDArray:
    """
    R = axis2R(angle, axis)

    Returns an SO3 object of the rotation specified by the axis-angle.

    :param float angle: the angle to rotate about the axis (in radians).
    :param NDArray axis: components of the unit axis about which to rotate as
        a numpy array [rx, ry, rz].
    :return R: 3x3 numpy array representing the rotation matrix.
    """
    # TODO fill this out
    R = np.array([
        [cos(angle) + axis[0]**2 * (1 - cos(angle)),
            axis[0] * axis[1] * (1 - cos(angle)) - axis[2] * sin(angle),
            axis[0] * axis[2] * (1 - cos(angle)) + axis[1] * sin(angle)],
        [axis[1] * axis[0] * (1 - cos(angle)) + axis[2] * sin(angle),
            cos(angle) + axis[1]**2 * (1 - cos(angle)),
            axis[1] * axis[2] * (1 - cos(angle)) - axis[0] * sin(angle)],
        [axis[2] * axis[0] * (1 - cos(angle)) - axis[1] * sin(angle),
            axis[2] * axis[1] * (1 - cos(angle)) + axis[0] * sin(angle),
            cos(angle) + axis[2]**2 * (1 - cos(angle))]
    ])
    return clean_rotation_matrix(R)


def R2quat(R: NDArray) -> NDArray:
    """
    quaternion = R2quat(R)

    Returns a quaternion representation of pose.

    :param NDArray R: 3x3 rotation matrix.
    :return quaternion: numpy array for the quaternion representation of pose in
        the format [nu, ex, ey, ez]
    """
    # TODO, see equation (2.34) and (2.35) on pg. 55, using functions like "sp.sqrt," and "sp.sign"
    nu = 0.5 * sp.sqrt(np.abs(np.trace(R) + 1))
    ex = (sp.sign(R[2, 1] - R[1, 2]) * sp.sqrt(R[0, 0] - R[1, 1] - R[2, 2] + 1)) / 2
    ey = (sp.sign(R[0, 2] - R[2, 0]) * sp.sqrt(R[1, 1] - R[2, 2] - R[0, 0] + 1)) / 2
    ez = (sp.sign(R[1, 0] - R[0, 1]) * sp.sqrt(R[2, 2] - R[0, 0] - R[1, 1] + 1)) / 2

    return np.array([nu, ex, ey, ez])

def quat2R(q: NDArray) -> NDArray:
    """
    R = quat2R(q)

    Returns a 3x3 rotation matrix from a quaternion.

    :param NDArray q: [nu, ex, ey, ez ] - defining the quaternion.
    :return R: numpy array, 3x3 rotation matrix.
    """
    # TODO, extract the entries of q below, and then calculate R
    nu = q[0]
    ex = q[1]
    ey = q[2]
    ez = q[3]
    R = np.array([[2*(nu**2+ex**2)-1,
                     2*(ex*ey-ez*nu),
                     2*(ex*ez+ey*nu)],

                     [2*(ex*ey+ez*nu),
                     2*(nu**2+ey**2)-1,
                     2*(ey*ez-ex*nu)],

                     [2*(ex*ez-ey*nu),
                     2*(ey*ez+ex*nu),
                     2*(nu**2+ez**2)-1]])
    return clean_rotation_matrix(R)


def euler2R(th1: float, th2: float, th3: float, order: str='xyz') -> NDArray:
    """
    R = euler2R(th1, th2, th3, order='xyz')

    Returns a 3x3 rotation matrix as specified by the euler angles, we assume in all cases
    that these are defined about the "current axis," which is why there are only 12 versions
    (instead of the 24 possiblities noted in the course slides).

    :param float th1: angle of rotation about 1st axis (rad)
    :param float th2: angle of rotation about 2nd axis (rad)
    :param float th3: angle of rotation about 3rd axis (rad)
    :param str order: specifies the euler rotation to use, for example 'xyx', 'zyz', etc.
    :return R: 3x3 numpy array, the rotation matrix.
    """

    # TODO - fill out each expression for R based on the condition
    # (hint: use your rotx, roty, rotz functions)
    if order == 'xyx':
        R = rotx(th1) @ roty(th2) @ rotx(th3)
    elif order == 'xyz':
        R = rotx(th1) @ roty(th2) @ rotz(th3)
    elif order == 'xzx':
        R = rotx(th1) @ rotz(th2) @ rotx(th3)
    elif order == 'xzy':
        R = rotx(th1) @ rotz(th2) @ roty(th3)
    elif order == 'yxy':
        R = roty(th1) @ rotx(th2) @ roty(th3)
    elif order == 'yxz':
        R = roty(th1) @ rotx(th2) @ rotz(th3)
    elif order == 'yzx':
        R = roty(th1) @ rotz(th2) @ rotx(th3)
    elif order == 'yzy':
        R = roty(th1) @ rotz(th2) @ roty(th3)
    elif order == 'zxy':
        R = rotz(th1) @ rotx(th2) @ roty(th3)
    elif order == 'zxz':
        R = rotz(th1) @ rotx(th2) @ rotz(th3)
    elif order == 'zyx':
        R = rotz(th1) @ roty(th2) @ rotx(th3)
    elif order == 'zyz':
        R = rotz(th1) @ roty(th2) @ rotz(th3)
    else:
        raise ValueError("Invalid Order!")

    return clean_rotation_matrix(R)

def R2euler(R, order='xyz'):

    D = dict(x=(rotx, 0), y=(roty, 1), z=(rotz, 2))

    rotA, axis1 = D[order[0]]
    rotB, axis2 = D[order[1]]
    rotC, axis3 = D[order[2]]

    if axis1 >= axis3:
        s = -1
    else:
        s = 1

    Ri = np.eye(3)
    Rf = R

    v = np.cross(Rf[:, axis3], (s * Ri[:, axis1]))
    if np.isclose(norm(v), 0):  # This indicates a rotation about the A axis ONLY.
        th1 = np.arccos(Ri[:, axis2] @ (Rf[:, axis2]))
        th2 = 0
        th3 = 0
        Ri = Ri @ rotA(th1)
    else:
        v = v / norm(v)
        th1 = np.arccos(min(max(Ri[:, axis2] @ v, 1), 0))
        Ri = Ri @ rotA(th1)

        th2 = np.arccos(min(max(Ri[:, axis3] @ Rf[:, axis3], 1), 0))
        Ri = Ri @ rotB(th2)

        th3 = np.arccos(min(max(Ri[:, axis2] @ Rf[:, axis2], 1), 0))
        Ri = Ri @ rotC(th3)

    return np.array([th1, th2, th3])