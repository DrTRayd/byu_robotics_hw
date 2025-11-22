import kinematics_fp as kin
import numpy as np
from visualization import VizScene

import transforms as tr

##Define a 6_DOF arm using DH parameters
dh_params = [[0,    1.0,   0.0,   np.pi/2.0],
             [0,    0.0,   1.0,   0],
             [0,    0.0,   1.0,   0], 
             [0,    0.0,   1.0,   0]]

tip = tr.se3(np.array([0, 0, 0]))
arm = kin.SerialArm(dh_params, jt = ['r', 'r', 'r', 'r'], tip=tip)

#List of desired end-effector positions and orientations
R = np.eye(3)
R[:,0] = np.array([1,0,0])
R[:,2] = np.array([0,0,-1])
R[:,1] = np.cross(R[:,2], R[:,0])  # ensures orthogonality
print(R)
desired_poses = [
    (np.array([1.0, 1.0, 0]), R),
    (np.array([1.5, 1.5, 0.0]), R),
    (np.array([2.0, 1.5, 0.0]), R)]

#Define q_0 and error
q_0 = np.array([0.0, 0.0, 0.0, 0.0])

#Logic vairables
i = 0
pos_error_thresh = 1e-3
orient_error_thresh = 1e-3

# Visualize current state
viz = VizScene()
viz.add_arm(arm)
viz.update(qs=[q_0])

#Iterate through each desired pose
for i in range(len(desired_poses)):
    pos_error = 1.0
    orient_error = 1.0
    # Continue until error is small enough
    while pos_error > pos_error_thresh and orient_error > orient_error_thresh:
        T_cur = arm.fk(q_0)

        # Build desired transform
        T_des = np.eye(4)
        T_des[0:3,0:3] = desired_poses[i][1] #orientation
        T_des[0:3,3]   = desired_poses[i][0] #position
        #print("T_des:")
        #print(T_des)

        delta_T = np.linalg.inv(T_cur) @ T_des

        axi_angle = tr.R2axis(delta_T[0:3,0:3])
        angle = axi_angle[0]
        axis = axi_angle[1:]
        z_cur = T_cur[0:3, 2]
        z_des = np.array([0,0,-1])
        rot_err = np.cross(z_cur, z_des)  # minimal rotation to align Z axes 
        print("Rotation Error:")
        print(rot_err)   
        # if np.isclose(angle, 0):
        #     rot_err = np.zeros(3)
        # else:
        #     rot_err = angle * axis
        pos_err = delta_T[0:3,3]
        #print("Position Error:")
        #print(pos_err)

        # Compute error magnitudes
        pos_error = np.linalg.norm(pos_err)
        orient_error = np.linalg.norm(rot_err)

        if pos_error < pos_error_thresh and orient_error < orient_error_thresh:
            break

        J = arm.jacob(q_0)

        # Form error (tool frame)
        error = np.hstack((pos_err, rot_err)).reshape(6,1)

        # Rotate into base frame
        R_n = T_cur[0:3,0:3]
        R_block = np.block([
            [R_n, np.zeros((3,3))],
            [np.zeros((3,3)), R_n]
        ])
        error_base = R_block @ error

        # Update rule
        K = 0.15 * np.eye(6)
        kd = 0.1  # damping factor, tweak for stability
        JJT = J @ J.T
        damped_inv = np.linalg.inv(JJT + kd**2 * np.eye(6))
        q_0 = q_0 + (J.T @ (damped_inv @ (K @ error_base))).flatten()
        print(q_0)

        # Update visualization
        viz.update(qs=[q_0])






# ##Define a 4_DOF arm using DH parameters
# dh_params = [[0,    1.0,   0.0,   np.pi/2.0],
#              [0,    0.0,   1.0,   0],
#              [0,    0.0,   1.0,   0], 
#              [0,    0.0,   1.0,   0]]
# tip = tr.se3(np.array([0, 0, 0]))
# arm = kin.SerialArm(dh_params, jt = ['r', 'r', 'r', 'r'], tip=tip)

# #List of desired end-effector positions and orientations
# R = np.eye(3)
# R[:,0] = np.array([1,0,0])
# R[:,2] = np.array([0,0,-1])
# R[:,1] = np.cross(R[:,2], R[:,0])  # ensures orthogonality
# print(R)
# desired_poses = [
#     (np.array([1.0, 1.0, 0]), R),
#     (np.array([1.5, 1.5, 0.0]), R),
#     (np.array([2.0, 1.5, 0.0]), R)]

# #Define q_0 and error
# q_0 = np.array([0.0, 0.0, 0.0, 0.0])

# #Logic vairables
# i = 0
# pos_error_thresh = 1e-3
# orient_error_thresh = 1e-3

# # Visualize current state
# viz = VizScene()
# viz.add_arm(arm)
# viz.update(qs=[q_0])

# #Iterate through each desired pose
# for i in range(len(desired_poses)):
#     pos_error = 1.0
#     orient_error = 1.0
#     # Continue until error is small enough
#     while pos_error > pos_error_thresh and orient_error > orient_error_thresh:
#         T_cur = arm.fk(q_0)

#         # Build desired transform
#         T_des = np.eye(4)
#         T_des[0:3,0:3] = desired_poses[i][1] #orientation
#         T_des[0:3,3]   = desired_poses[i][0] #position

#         delta_T = np.linalg.inv(T_cur) @ T_des

#         axi_angle = tr.R2axis(delta_T[0:3,0:3])
#         angle = axi_angle[0]
#         axis = axi_angle[1:]
#         z_cur = T_cur[0:3, 2]
#         z_des = np.array([0,0,-1])
#         rot_err = np.cross(z_cur, z_des)  # minimal rotation to align Z axes    
#         # if np.isclose(angle, 0):
#         #     rot_err = np.zeros(3)
#         # else:
#         #     rot_err = angle * axis
#         pos_err = delta_T[0:3,3]
#         print("Position Error:")
#         print(pos_err)

#         # Compute error magnitudes
#         pos_error = np.linalg.norm(pos_err)
#         orient_error = np.linalg.norm(rot_err)

#         if pos_error < pos_error_thresh and orient_error < orient_error_thresh:
#             break

#         J = arm.jacob(q_0)

#         # Form error (tool frame)
#         error = np.hstack((pos_err, rot_err)).reshape(6,1)

#         # Rotate into base frame
#         R_n = T_cur[0:3,0:3]
#         R_block = np.block([
#             [R_n, np.zeros((3,3))],
#             [np.zeros((3,3)), R_n]
#         ])
#         error_base = R_block @ error

#         # Update rule
#         K = 0.15 * np.eye(6)
#         q_0 = q_0 + (J.T @ (K @ error_base)).flatten()
#         print(q_0)

#         # Update visualization
#         viz.update(qs=[q_0])