import numpy as np
import kinematics
from visualization import VizScene
import sympy as sp
import time


q_initial_1 = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
q_initial_2 = np.array([np.pi/4, np.pi/4, np.pi/4, np.pi/4, np.pi/4, np.pi/4])


des_pos_1 = np.array([-0.149, 0.364, 1.03])
des_pos_2 = np.array([-0.171, -0.682, -0.192])
des_pos_3 = np.array([0.822, -0.1878, 0.533])
des_pos_4 = np.array([-0.336, 0.095, 0.931])
des_pos_5 = np.array([0.335, 0.368, 0.88])

#Create the arm

dh = np.array([[0, 0.2, 0, -np.pi/2],
                [0, 0, 0.2, 0],
                [np.pi/2, 0, 0, np.pi/2],
                [np.pi/2, 0.4, 0, -np.pi/2],
                [0, 0, 0, np.pi/2],
                [0, 0.4, 0, 0]])

jt_types = np.array(['r', 'r', 'r', 'r', 'r', 'r'])

arm = kinematics.SerialArm(dh, jt=jt_types)

K = np.eye(3)*1.0

#Problem 3 Jacobian transpose method for IK

def jacobian_transpose_ik(arm, q_initial, des_pos, alpha=0.1, error_minimum=1e-3, max_iter=1000):
    q_start = q_initial.copy()
    for i in range(max_iter):

        T = arm.fk(q_start)
        current_pos = T[0:3, 3]
        error = des_pos - current_pos

        if np.linalg.norm(error) < error_minimum:
            print(f"Converged in {i} iterations.")
            return q_start
        
        J = arm.jacob(q_start)
        J_pos = J[0:3, :]  # Extract position part of Jacobian
        J_transpose = J_pos.T

        delta = J_transpose @ (K @ error)
        q_start += delta


q_sol_1 = jacobian_transpose_ik(arm, q_initial_1, des_pos_1)
q_sol_2 = jacobian_transpose_ik(arm, q_initial_2, des_pos_2)
q_sol_3 = jacobian_transpose_ik(arm, q_initial_1, des_pos_3)
q_sol_4 = jacobian_transpose_ik(arm, q_initial_2, des_pos_4)
q_sol_5 = jacobian_transpose_ik(arm, q_initial_1, des_pos_5)

q_sol_1_2 = jacobian_transpose_ik(arm, q_initial_2, des_pos_1)
q_sol_2_2 = jacobian_transpose_ik(arm, q_initial_2, des_pos_2)
q_sol_3_2 = jacobian_transpose_ik(arm, q_initial_2, des_pos_3)
q_sol_4_2 = jacobian_transpose_ik(arm, q_initial_2, des_pos_4)
q_sol_5_2 = jacobian_transpose_ik(arm, q_initial_2, des_pos_5)

print("Solution 1 (Starting Position 1):", q_sol_1)
print("Solution 2 (Starting Position 1):", q_sol_2)
print("Solution 3 (Starting Position 1):", q_sol_3)
print("Solution 4 (Starting Position 1):", q_sol_4)
print("Solution 5 (Starting Position 1):", q_sol_5)

print("Solution 1 (Starting Position 2):", q_sol_1_2)
print("Solution 2 (Starting Position 2):", q_sol_2_2)
print("Solution 3 (Starting Position 2):", q_sol_3_2)
print("Solution 4 (Starting Position 2):", q_sol_4_2)
print("Solution 5 (Starting Position 2):", q_sol_5_2)

# Make plot of original position and desired position one

viz = VizScene()
viz.add_arm(arm)
viz.add_marker(des_pos_1)
viz.update(qs=[q_sol_1])
viz.hold()

