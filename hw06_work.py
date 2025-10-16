import numpy as np
import sympy as sp
import kinematics
from visualization import VizScene


# # Problem 1


# # find jacobian for the 2 link manipulator

# dh_params = np.array([[0, 0, 1, -np.pi/2],
#              [0, 0, 1, 0]])
# jt_types = np.array(['r', 'r'])

# arm = kinematics.SerialArm(dh_params, jt=jt_types)

# q_set = np.array([0, 0])
# force = np.array([1, 0, 0, 0, 0, 0])

# J = arm.jacob(q_set)
# print(J)

# # find torque at each joint
# tau = J.T @ force
# print(tau)


# # Problem 2

# define dh parameters for 6 DOF arm
dh = np.array([[0, 0.2, 0, -np.pi/2],
                [0, 0, 0.2, 0],
                [np.pi/2, 0, 0, np.pi/2],
                [np.pi/2, 0.4, 0, -np.pi/2],
                [0, 0, 0, np.pi/2],
                [0, 0.4, 0, 0]])

jt_types = np.array(['r', 'r', 'r', 'r', 'r', 'r'])

arm = kinematics.SerialArm(dh, jt=jt_types)

#calculate forward kinematics for arm
T = arm.fk([np.pi/2, 0, np.pi/4, 0, np.pi/2, 0])
print(T)

viz = VizScene()
viz.add_arm(arm)
q = [0,0,0,0,0,0]
q_new = [np.pi/2, 0, np.pi/4, 0, np.pi/2, 0]
viz.update(qs=[q_new])

# add goal at tip of arm
goal = T[0:3, 3]
viz.add_marker(goal)
viz.hold()

