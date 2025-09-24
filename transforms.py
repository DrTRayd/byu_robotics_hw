"""
Transforms Module - Contains code for to learn about rotations
and eventually homogenous transforms.

Empty outline derived from code written by John Morrell, former TA.
"""

import numpy as np
from numpy import sin, cos, sqrt
from numpy.typing import NDArray
from utility import clean_rotation_matrix


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
