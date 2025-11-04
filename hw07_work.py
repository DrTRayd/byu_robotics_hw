
# Problem 1
from visualization import ArmPlayer
from kinematics import SerialArm
import numpy as np
import sympy as sp
import transforms as tf

# these DH parameters are based on solutions from HW 3, if you
# pick a different set that still describe the robots accurately,
# that's great.
# a_len = 0.5
# d_len = 0.35

# dh_part_a = [[0, d_len, 0., np.pi/2.0],
#             [0, 0, a_len, 0], 
#             [0, 0, a_len, 0]]

# dh_part_b = [[0, d_len, 0., -np.pi/2.0],
#             [0, 0, a_len, 0], 
#             [np.pi/2.0, 0, 0, np.pi/2.0], 
#             [np.pi/2.0, d_len*2, 0, -np.pi/2.0],
#             [0, 0, 0, np.pi/2],
#             [0, d_len*2, 0, 0]]

# jt_types_a = ['r', 'r', 'r']
# jt_types_b = ['r', 'r', 'r', 'r', 'r', 'r']
# arm = SerialArm(dh_part_b, jt=jt_types_b)
# ArmPlayer(arm)

# # Calulate Jacobian at zero position
# J = arm.jacob(np.zeros(arm.n))
# print("Jacobian at zero position for Part A:\n", J)
# print(" the rank is:", np.linalg.matrix_rank(J))
# print("the condition number is:", np.linalg.cond(J))

# #Calculate when Jacobian loses determinant of J is zero
# q_sing = [0, 0, 0, 0, 0, 0]
# J_sing = arm.jacob(q_sing)
# Jv = J_sing[0:3, :]
# print("Jacobian at singular configuration for Part A:\n", J_sing)
# print(" the rank of translation is:", np.linalg.matrix_rank(Jv))
# print("the condition number is:", np.linalg.cond(J_sing))


## Problem 3

# # Define symoblic variables
# s_theta1, c_theta1, s_theta2, c_theta2 = sp.symbols('s_theta1 c_theta1 s_theta2 c_theta2')
# # Define original jacobian

# J = sp.Matrix([[-s_theta1*c_theta2 - s_theta1, -s_theta2*c_theta1],
#                 [c_theta1*c_theta2 + c_theta1, -s_theta2*s_theta1],
#                 [0, -s_theta1*s_theta1*c_theta2 - c_theta1*c_theta1*c_theta2], 
#                 [0, -s_theta1],
#                 [0, c_theta1],
#                 [1, 0]])

# #Compute rotation matric from end-effector to base frame 

# R_b_ee = tf.sp_rotx(sp.pi/2)

# print("Rotation Matrix from end-effector to base frame:\n", R_b_ee)
# R_b_ee = sp.Matrix([[1, 0, 0, 0, 0, 0], 
#                     [0, 0, -1, 0, 0, 0],
#                     [0, 1, 0, 0, 0, 0],
#                     [0, 0, 0, 1, 0, 0],
#                     [0, 0, 0, 0, 0, -1],
#                     [0, 0, 0, 0, 1, 0]])



# # Solve for transformed Jacobian
# Z = R_b_ee * sp.Matrix([[1, 0, 0, 0, 0, 0],
#                         [0, 1, 0, 0, 0, 2],
#                         [0, 0, 1, 0, 2, 0],
#                         [0, 0, 0, 1, 0, 0],
#                         [0, 0, 0, 0, 1, 0],
#                         [0, 0, 0, 0, 0, 1]])

# sp.pprint(Z)

# J_transformed = Z * J
# sp.pprint(J_transformed)

## Problem 4

dh = np.array([[0, 0, 0, np.pi/2],
               [0, 0, 0.4318, 0],
                [0, 0.15, 0.02, -np.pi/2],
                [0, 0.4318, 0, np.pi/2],
                [0, 0, 0, -np.pi/2],
                [0, 0.4, 0, 0]])

jt_types = ['r', 'r', 'r', 'r', 'r', 'r']
arm = SerialArm(dh, jt=jt_types)

T_tool_6 = np.array([[0, 0, 1, 0], 
                     [0, 1, 0, 0], 
                     [-1, 0, 0, 0.2], 
                     [0, 0, 0, 1]])

#Find jacobian at the tool tip an in the tool frame using shifting law
