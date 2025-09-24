import transforms as tr
import numpy as np
import sympy as sp
from visualization import VizScene


## Problem 2-37

# find the transformation from frame 0 to 3
# T3_in_0 = tr.se3(tr.rotz(np.pi/2)@tr.rotx(np.pi), [-0.5, 1.5, 3])

# T1_in_0 = tr.se3(p=[0, 1, 1])
# T2_in_0 = tr.se3(p=[-0.5, 1.5, 1])

# print("T1_in_0:", T1_in_0)
# print("T2_in_0:", T2_in_0)
# print("T3_in_0:", T3_in_0)

# # Plots frames
# viz = VizScene()
# viz.add_frame(np.eye(4), label='world', axes_label='w')
# viz.add_frame(T1_in_0, label='frame1', axes_label='1')
# viz.add_frame(T2_in_0, label='frame2', axes_label='2')
# viz.add_frame(T3_in_0, label='frame3', axes_label='3')
# viz.hold()  # holds the visualization

## Problem 2-38
# Rotate about z again by pi/2

# T3_in_0_new = tr.se3(tr.rotz(np.pi/2)@tr.rotx(np.pi)@tr.rotz(np.pi/2), [-0.5, 1.5, 3])

# print("T3_in_0_new:", T3_in_0_new)

## Problem 3-4

#Solve for the A matrices per D-H convention
# q1, q3, d1, d3, a0 = sp.symbols('q1 q3 d1 d3 a0')


# Rot_1_z = tr.sp_se3(R=tr.sp_rotz(q1))
# Trans_1_z = tr.sp_se3(p=[a0,0,0])
# Trans_1_x = tr.sp_se3(p=[0,0,0])
# Rot_1_x = tr.sp_se3(R=tr.sp_rotx(-np.pi/2))

# Rot_2_z = tr.sp_se3(R=tr.sp_rotz(np.pi/2))
# Trans_2_x = tr.sp_se3(p=[0,0,d1])
# Trans_2_z = tr.sp_se3( p=[0,0,0])
# Rot_2_x = tr.sp_se3(R=tr.sp_rotx(np.pi/2))

# Rot_3_z = tr.sp_se3(R=tr.sp_rotz(0))
# Trans_3_x = tr.sp_se3(p=[0,0,0])
# Trans_3_z = tr.sp_se3( p=[0,0,q3])
# Rot_3_x = tr.sp_se3(R=tr.sp_rotx(0))

# A1 = Rot_1_z @ Trans_1_z @ Trans_1_x @ Rot_1_x
# A2 = Rot_2_z @ Trans_2_z @ Trans_2_x @ Rot_2_x
# A3 = Rot_3_z @ Trans_3_z @ Trans_3_x @ Rot_3_x

# print("A1:")
# sp.pprint(sp.simplify(A1))
# print("A2:")
# sp.pprint(sp.simplify(A2))
# print("A3:")
# sp.pprint(sp.simplify(A3))

# FK = sp.simplify(A1 @ A2 @ A3)
# print("FK:")
# sp.pprint(FK)

## Problem 3-6

#Solve for the A matrices per D-H convention
# d1, a2, a3 = sp.symbols('d1 a2 a3')


# Rot_1_z = tr.sp_se3(R=tr.sp_rotz(0))
# Trans_1_z = tr.sp_se3(p=[0,0,a2])
# Trans_1_x = tr.sp_se3(p=[0,0,0])
# Rot_1_x = tr.sp_se3(R=tr.sp_rotx(np.pi/2))

# Rot_2_z = tr.sp_se3(R=tr.sp_rotz(0))
# Trans_2_z = tr.sp_se3(p=[0,0,0])
# Trans_2_x = tr.sp_se3(p=[a2,0,0])
# Rot_2_x = tr.sp_se3(R=tr.sp_rotx(0))

# Rot_3_z = tr.sp_se3(R=tr.sp_rotz(0))
# Trans_3_z = tr.sp_se3(p=[0,0,0])
# Trans_3_x = tr.sp_se3( p=[a3,0,0])
# Rot_3_x = tr.sp_se3(R=tr.sp_rotx(0))

# A1 = Rot_1_z @ Trans_1_z @ Trans_1_x @ Rot_1_x
# A2 = Rot_2_z @ Trans_2_z @ Trans_2_x @ Rot_2_x
# A3 = Rot_3_z @ Trans_3_z @ Trans_3_x @ Rot_3_x

# print("A1:")
# sp.pprint(sp.simplify(A1))
# print("A2:")
# sp.pprint(sp.simplify(A2))
# print("A3:")
# sp.pprint(sp.simplify(A3))

# FK = sp.simplify(A1 @ A2 @ A3)
# print("FK:")
# sp.pprint(FK)

# Part g

d1, a2, a3 = sp.symbols('d1 a2 a3')

Rot_1_z = tr.sp_se3(R=tr.sp_rotz(np.pi/2))
Trans_1_z = tr.sp_se3(p=[0,0,d1])
Trans_1_x = tr.sp_se3(p=[0,0,0])
Rot_1_x = tr.sp_se3(R=tr.sp_rotx(np.pi))

Trans_2_y = tr.sp_se3(p=[0,-a2,0])
Rot_2_x = tr.sp_se3(R=tr.sp_rotx(np.pi/2))
Rot_2_z = tr.sp_se3(R=tr.sp_rotz(-np.pi/2))

Trans_3_z = tr.sp_se3(p=[0,0,a3])
Rot_3_x = tr.sp_se3(R=tr.sp_rotx(-np.pi/2))
Rot_3_z = tr.sp_se3(R=tr.sp_rotz(-np.pi/2))

A1 = Rot_1_z @ Trans_1_z @ Trans_1_x @ Rot_1_x
A2 = Trans_2_y @ Rot_2_x @ Rot_2_z
A3 = Trans_3_z @ Rot_3_x @ Rot_3_z

print("A1:")
sp.pprint(sp.simplify(A1))
print("A2:")
sp.pprint(sp.simplify(A2))
print("A3:")
sp.pprint(sp.simplify(A3))

FK = sp.simplify(A1 @ A2 @ A3)
print("FK:")
sp.pprint(FK)