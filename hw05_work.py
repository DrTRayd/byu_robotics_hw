import transforms as tr
import sympy as sp
import numpy as np
import kinematics as kin
from visualization import VizScene

# problem 2a
# q1, q2, d1, a2 = sp.symbols('q1 q2 d1 a2')

# A1 = sp.Matrix([[sp.cos(q1), 0, sp.sin(q1), 0],
#                 [sp.sin(q1), 0, -sp.cos(q1), 0],
#                 [0, 1, 0, d1],
#                 [0, 0, 0, 1]])

# A2 = sp.Matrix([[sp.cos(q2), -sp.sin(q2), 0, a2*sp.cos(q2)],
#                 [sp.sin(q2), sp.cos(q2), 0, a2*sp.sin(q2)],
#                 [0, 0, 1, 0],
#                 [0, 0, 0, 1]])

# O_2_0 = A1 @ A2
# O_1_0 = A1

# print("O_2_0 = ")
# sp.pprint(O_2_0)
# print("\nO_1_0 = ")
# sp.pprint(O_1_0)

# r_02_0 = O_2_0
# r_12_0 = O_2_0 - O_1_0

# top_left = sp.Matrix([0, 0, 1]).cross(r_02_0[0:3, 3])
# top_right = sp.Matrix([0, -1, 0]).cross(r_12_0[0:3, 3])

# print("\nTop left part of Jacobian:")
# sp.pprint(top_left)
# print("\nTop right part of Jacobian:")
# sp.pprint(top_right)

# Problem 2b
# Generating a serial arm to represent the robot in the problem statement



# dh = [[np.pi/2, 30, 0, np.pi/2],
#       [0, 0, 30, 0]]
# jt_types = ['r', 'r']
# serial_arm = kin.SerialArm(dh, jt=jt_types)

# J = serial_arm.jacob([0, 0])
# J = sp.Matrix(J)
# print("Jacobian for problem 2b:")
# sp.pprint(sp.nsimplify(J))

## Problem 3
# Compute the forward kinematics of a robot arm and its Jacobian symbolically.
# q1, q2, a1, ac = sp.symbols('q1 q2 a1 ac')

# r1 = tr.sp_rotz(q1)
# p1 = sp.Matrix([a1, 0, 0])

# # First symbolic transformation matrix
# A1 = sp.eye(4)
# A1[:3, :3] = r1
# A1[:3, 3] = p1

# r2 = tr.sp_rotz(q2)
# p2 = sp.Matrix([ac, 0, 0])

# # Second symbolic transformation matrix
# A2 = sp.eye(4)
# A2[:3, :3] = r2
# A2[:3, 3] = p2

# T2_0 = A1 @ A2

# print("T2_0 = ")
# sp.pprint(sp.simplify(T2_0))

# # Jacobian calculation
# z0 = sp.Matrix([0, 0, 1])

# top_left = z0.cross(T2_0[:3, 3])
# top_right = A1[:3, 2].cross(T2_0[:3, 3] - A1[:3, 3])

# print("\nTop left part of Jacobian:")
# sp.pprint(sp.simplify(top_left))
# print("\nTop right part of Jacobian:")
# sp.pprint(sp.simplify(top_right))


## Problem 4
dh = [[0, 0, 0, -np.pi/2.0],
      [0, 0.154, 0, np.pi/2.0],
      [0, 0, 0.25,0],
      [-np.pi/2, 0, 0.0, -np.pi/2.0],
      [-np.pi/2, 0, 0.0, np.pi/2.0],
      [np.pi/2, 0.263, 0, 0.0]]

jt_types = ['r', 'r', 'p', 'r', 'r', 'r']
tip = np.eye(4)
tip[:3, :3] = tr.roty(-np.pi/2)
arm = kin.SerialArm(dh, jt=jt_types, tip=tip)

q_default = [0, 0, 0.1, 0, 0, 0]

viz = VizScene()

viz.add_arm(arm, draw_frames=True)
viz.update(qs=[q_default])


viz.hold()

# calculate jacobian
J = arm.jacob(q_default)
J = sp.Matrix(J)
print("Jacobian:")
sp.pprint(sp.nsimplify(J))