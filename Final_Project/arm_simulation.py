import kinematics_fp as kin
import numpy as np

import transforms_fp as tr

##Define a 4_DOF arm using DH parameters
dh_params = [[0,    1.0,   0.0,   np.pi/2.0],
             [0,    0.0,   1.0,   0],
             [0,    0.0,   1.0,   0], 
             [0,    0.0,   0.5,   0]]

arm = kin.SerialArm(dh_params, jt = ['r', 'r', 'r', 'r'])

#List of desired end-effector positions and orientations
downward_orientation = tr.rotx(np.pi)
desired_poses = [
    (np.array([1.5, 0.0, 1.5]), downward_orientation),
    (np.array([1.0, 1.0, 1.0]), downward_orientation),
    (np.array([0.5, -1.0, 0.5]), downward_orientation)]

#Define q_0 and error
q_0 = np.array([0.0, 0.0, 0.0, 0.0])

#Logic vairables
i = 0
pos_error_thresh = 1e-3
orient_error_thresh = 1e-3

# Visualize current state
viz_scene = viz.VizScene()
viz_scene.add_arm(arm, q_0)
viz_scene.show()

#Iterate through each desired pose
for i in range(len(desired_poses)):
    while True:
        T_cur = arm.fk(q_0)

        # Build desired transform
        T_des = np.eye(4)
        T_des[0:3,0:3] = desired_poses[i][1] #orientation
        T_des[0:3,3]   = desired_poses[i][0] #position

        delta_T = np.linalg.inv(T_cur) @ T_des

        ang, vec = tr.R2axis(delta_T[0:3,0:3])
        rot_err = ang * vec
        pos_err = delta_T[0:3,3]

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
        K = 0.5 * np.eye(6)
        q_0 = q_0 + (J.T @ (K @ error_base)).flatten()

        # Update visualization
        viz.update(qs=[q_0])