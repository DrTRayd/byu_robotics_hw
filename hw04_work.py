import transforms as tr
import numpy as np
import sympy as sp
import kinematics as km
from visualization import VizScene

# R = tr.roty(90*np.pi/180.0) @ tr.rotz(45*np.pi/180.0)

# T = tr.se3(R=R, p=[0, 0, 0])

# axis_angle = tr.R2axis(R)
# print("Axis/angle representation of R:")
# print(axis_angle)

# # Show initial and final frames of rotation
# viz = VizScene()
# viz.add_frame(np.eye(4), label="I")
# viz.add_frame(T, label="R")
# viz.hold()

# # Find equivalent quaternion
# quat = tr.R2quat(R)
# print("Quaternion representation of R:")
# print(quat)



## Part 2: 
psi, theta, phi = sp.symbols('psi theta phi')

# psi = yaw
# theta = pitch
# phi = roll

R_rpy = tr.sp_rotz(psi) @ tr.sp_roty(theta) @ tr.sp_rotx(phi)

print("R from rpy angles:")
sp.pprint(sp.simplify(R_rpy))

#Transform in the z - direction 

unit_z = sp.Matrix([0, 0, 1])

sp.pprint(R_rpy @ unit_z)



## Algoritm to convert from R to roll pitch yaw
def R2rpy(R: sp.Matrix) -> sp.Matrix:
    """
    Convert a rotation matrix to roll, pitch, yaw angles.
    Args:
        R: A 3x3 sympy Matrix representing a rotation matrix.
    Returns:
        A 3x1 sympy Matrix representing the roll, pitch, yaw angles.
    """

    roll = sp.atan2(R[2, 1], R[2, 2])
    pitch = sp.atan2(-R[2, 0], sp.sqrt(R[2, 1]**2 + R[2, 2]**2))
    yaw = sp.atan2(R[1, 0], R[0, 0])

    return sp.Matrix([roll, pitch, yaw])

print("R2rpy of R_rpy:")
sp.pprint(sp.simplify(R2rpy(R_rpy)))
