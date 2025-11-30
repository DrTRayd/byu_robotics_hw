import kinematics_fp as kin
import numpy as np
from visualization import VizScene
import transforms as tr

# -----------------------------
# Define 6-DOF arm (full 3 position + 3 orientation control)
# -----------------------------
# 6-DOF provides complete control over end effector position and orientation
dh_params = [
    [0, 1.0, 0.0, np.pi/2.0],  # Joint 1: base rotation (vertical axis)
    [0, 0.0, 1.0, 0],          # Joint 2: shoulder (horizontal link)
    [0, 0.0, 1.0, 0],          # Joint 3: elbow (horizontal link)
    [0, 0.0, 0.0, np.pi/2.0],  # Joint 4: wrist roll
    [0, 0.0, 0.0, -np.pi/2.0], # Joint 5: wrist pitch
    [0, 0.0, 0.5, 0]           # Joint 6: wrist yaw (tool frame)
]

# Tip frame: rotate so the tool's Z-axis points down when arm is at zero config
# Rotating 180° around X makes Z point down
tip_rot = tr.rotx(np.pi)  # Z will point in -Z direction
tip = tr.se3(tip_rot, np.array([0, 0, 0]))

arm = kin.SerialArm(dh_params, jt=['r']*6, tip=tip)

# Desired poses - end effector pointing straight down
# For Z pointing down, X pointing forward: right-handed frame
z_des = np.array([0, 0, -1])
x_des = np.array([1, 0, 0])
y_des = np.cross(z_des, x_des)  # This gives [0, 1, 0] which is correct
R_des = np.column_stack([x_des, y_des, z_des])

desired_poses = [
    (np.array([1.0, 0.0, 1.0]), R_des),   # Reachable positions for 6-DOF
    (np.array([1.5, 0.5, 1.5]), R_des),
    (np.array([1.0, 1.0, 2.0]), R_des)
]

# Initial joint configuration - start with a reasonable pose for 6-DOF
q_0 = np.array([0.0, np.pi/6, np.pi/6, 0.0, -np.pi/3, 0.0])

# Visualization
viz = VizScene()
viz.add_arm(arm)
viz.update(qs=[q_0])

# Error thresholds
pos_error_thresh = 5e-3   # Relaxed position tolerance
orient_error_thresh = 5e-3  # Relaxed orientation tolerance

# -----------------------------
# Iteration loop
# -----------------------------
for i in range(len(desired_poses)):
    p_des, R_des = desired_poses[i]
    
    max_iterations = 1000
    iteration = 0
    
    while iteration < max_iterations:
        iteration += 1
        # Current end-effector pose
        T_cur = arm.fk(q_0, tip=True)
        R_cur = T_cur[:3,:3]
        p_cur = T_cur[:3,3]

        # -----------------------------
        # Position error (3x1)
        # -----------------------------
        pos_err = p_des - p_cur

        # -----------------------------
        # Orientation error - prioritize Z pointing down
        # Use axis-angle representation for better convergence
        # -----------------------------
        # Compute rotation error matrix: R_error = R_desired * R_current^T
        R_err = R_des @ R_cur.T
        
        # Convert to axis-angle to get orientation error vector
        axis_angle = tr.R2axis(R_err)
        angle = axis_angle[0]
        axis = axis_angle[1:]
        
        # Full 3D orientation error vector (in world frame)
        if np.abs(angle) < 1e-6:
            ori_err_full = np.zeros(3)
        else:
            ori_err_full = angle * axis
        
        # For 5-DOF: we want all position control + Z-axis orientation
        # The Z-axis error is most important for pointing down
        # We can use a weighted error or project onto controllable subspace
        
        # Use full 6D error with adaptive weighting
        error_6d = np.hstack((pos_err, ori_err_full))
        
        # Adaptive weight matrix: prioritize Z-alignment more when position is close
        pos_weight = 2.0 if np.linalg.norm(pos_err) > 0.1 else 1.0
        ori_weight = 1.0 if np.linalg.norm(pos_err) > 0.1 else 2.0
        W = np.diag([pos_weight, pos_weight, pos_weight, ori_weight, ori_weight, ori_weight])
        weighted_error = W @ error_6d

        # Print debug
        if iteration == 1 or iteration % 50 == 0:
            z_cur = R_cur[:, 2]
            z_des = R_des[:, 2]
            print(f"Pose {i+1}, Iter {iteration}: Pos Err: {np.linalg.norm(pos_err):.6f}, Ori Err: {np.linalg.norm(ori_err_full):.6f}")
            print(f"  Z_cur: [{z_cur[0]:.3f}, {z_cur[1]:.3f}, {z_cur[2]:.3f}], Z_des: [{z_des[0]:.3f}, {z_des[1]:.3f}, {z_des[2]:.3f}]")
            print(f"  Dot product (alignment): {np.dot(z_cur, z_des):.3f}")

        # Stop condition - check both position and orientation alignment
        z_cur = R_cur[:, 2]
        z_des = R_des[:, 2]
        z_alignment = np.dot(z_cur, z_des)  # Should be close to 1.0 for aligned
        
        # Check convergence: position close AND Z-axis aligned (dot product close to 1)
        if np.linalg.norm(pos_err) < pos_error_thresh and abs(1.0 - z_alignment) < orient_error_thresh:
            print(f"  [OK] Converged in {iteration} iterations! Z-alignment: {z_alignment:.6f}")
            break
        
        if iteration >= max_iterations:
            print(f"  [FAIL] Max iterations reached. Final errors: pos={np.linalg.norm(pos_err):.6f}, Z-align={z_alignment:.6f}")
            break

        # -----------------------------
        # Jacobian and DLS update
        # -----------------------------
        J_full = arm.jacob(q_0, tip=True)      # 6x6 for 6-DOF - include tip transform
        
        # Damped Least Squares with weighting
        # For 6-DOF arm with 6D task: square Jacobian, well-posed problem
        kd = 0.05  # Damping for stability near singularities
        
        # Adaptive step size based on error magnitude
        error_mag = np.linalg.norm(error_6d)
        if error_mag > 1.0:
            step_size = 0.3  # Smaller steps for large errors
        elif error_mag > 0.1:
            step_size = 0.5  # Medium steps
        else:
            step_size = 0.7  # Larger steps when close
        
        # Weighted damped least squares: minimize ||J*dq - error||_W^2 + kd^2*||dq||^2
        # Solution: dq = J^T * (J*J^T + kd^2*I)^-1 * W * error
        JJT = J_full @ J_full.T
        dq = J_full.T @ np.linalg.inv(JJT + kd**2 * np.eye(6)) @ weighted_error

        # Update joint angles
        q_0 = q_0 + step_size * dq
        
        # Joint limit safety (prevents wild motions)
        q_0 = np.clip(q_0, -np.pi, np.pi)

        # Update visualization
        viz.update(qs=[q_0])


# import kinematics_fp as kin
# import numpy as np
# from visualization import VizScene

# import transforms as tr

# ##Define a 5_DOF arm using DH parameters (3 position + 2 orientation control)
# dh_params = [[0,    1.0,   0.0,   np.pi/2.0],
#              [0,    0.0,   1.0,   0],
#              [0,    0.0,   1.0,   0], 
#              [0,    0.0,   1.0,   0], 
#              [0,    0.0,   1.0,   0]]

# # For example, rotate last frame so Z points down
# tip_rot = tr.axis2R(angle=0, axis=[0,0,1])  # example rotation
# tip = tr.se3(tip_rot[:3, :3])  # if tr.se3 accepts rotation matrix only

# arm = kin.SerialArm(dh_params, jt=['r', 'r', 'r', 'r', 'r'], tip=tip)


# #List of desired end-effector positions and orientations
# z_des = np.array([0,0,-1])
# x_des = np.array([1,0,0])
# y_des = np.cross(x_des, z_des) 
# R_des = np.column_stack([x_des, y_des, z_des])


# desired_poses = [
#     (np.array([1.5, 1.5, -0.5]), R_des),
#     (np.array([1.5, 1.5, 0.0]), R_des),
#     (np.array([2.0, 1.5, 0.0]), R_des)]

# #Define q_0 and error
# q_0 = np.array([0.0, np.pi/2, 0.0, 0.0, 0.0])

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
#     while True:
#         p_des = desired_poses[i][0]
#         R_des = desired_poses[i][1]

#         T_cur = arm.fk(q_0)
#         R_cur = T_cur[:3,:3]
#         p_cur = T_cur[:3,3]

#         # Position error (3x1)
#         pos_err = p_des - p_cur

#         # Simplified orientation error: align tool Z with desired Z
#         z_cur = R_cur[:,2]
#         z_des = R_des[:,2]
#         rot_vec = np.cross(z_cur, z_des)      # 3x1
#         ori_err = rot_vec[2]                  # pick ONE axis (4 DOF total)

#         # 4×1 total error
#         error = np.hstack((pos_err, [ori_err]))
#         print("Position Error Norm:", np.linalg.norm(pos_err), "Orientation Error:", ori_err)

#         # Stop condition
#         if np.linalg.norm(pos_err) < pos_error_thresh and abs(ori_err) < orient_error_thresh:
#             break

#         # Jacobian (6×4)
#         J = arm.jacob(q_0)

#         # Select only the rows corresponding to your 4 controlled dimensions
#         J_reduced = np.vstack([J[0], J[1], J[2], J[3]])   # 4×4
#         # or if blue = rotation around x or z etc.

#         # Damped least squares update
#         kd = 0.1
#         dq = J_reduced.T @ np.linalg.inv(J_reduced @ J_reduced.T + kd**2 * np.eye(4)) @ error

#         q_0 = q_0 + 0.1 * dq

#         viz.update(qs=[q_0])


# for i in range(len(desired_poses)):
#     pos_error = 1.0
#     orient_error = 1.0
#     # Continue until error is small enough
#     while pos_error > pos_error_thresh and orient_error > orient_error_thresh:
#         T_cur = arm.fk(q_0)

#         # Desired pose
#         R_des = desired_poses[i][1]
#         p_des = desired_poses[i][0]

#         # Current pose
#         R_cur = T_cur[0:3, 0:3]
#         p_cur = T_cur[0:3, 3]

#         # Position error (already world-frame)
#         pos_err = p_des - p_cur

#         # Orientation error (Eq. 3.84: Rd * Re^T)
#         R_err = R_des @ R_cur.T

#         # Convert rotation matrix to axis-angle
#         angle_axis = tr.R2axis(R_err)
#         angle = angle_axis[0]
#         ax = angle_axis[1:]
#         # Rotation error vector (world-frame twist)
#         rot_err = np.cross(R_cur[:,2], R_des[:,2])
#         ori_err_scalar = rot_err[0] 
        

#         # print("Rotation Error:")
#         # print(rot_err)   
#         # if np.isclose(angle, 0):
#         #     rot_err = np.zeros(3)
#         # else:
#         #     rot_err = angle * axis

#         # pos_err = delta_T[0:3,3]

#         # print("Position Error:")
#         # print(pos_err)

#         # Compute error magnitudes
#         pos_error = np.linalg.norm(pos_err)
#         orient_error = np.linalg.norm(rot_err)

#         if pos_error < pos_error_thresh and orient_error < orient_error_thresh:
#             break

#         J = arm.jacob(q_0)

#         # Form error (tool frame)
#         # error = np.hstack((pos_err, rot_err)).reshape(6,1)
#         error = np.hstack((pos_err, [ori_err_scalar]))

#         # Rotate into base frame
#         R_n = T_cur[0:3,0:3]
#         R_block = np.block([
#             [R_n, np.zeros((3,3))],
#             [np.zeros((3,3)), R_n]
#         ])
#         error_base = R_block @ error

#         # Update rule
#         K = 0.015 * np.eye(6)
#         kd = 0.1  # damping factor, tweak for stability
#         JJT = J @ J.T
#         damped_inv = np.linalg.inv(JJT + kd**2 * np.eye(6))
#         q_0 = q_0 + (J.T @ (damped_inv @ (K @ error_base))).flatten()
#         # print(q_0)

#         # Update visualization
#         viz.update(qs=[q_0])






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