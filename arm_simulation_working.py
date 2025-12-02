import kinematics_fp as kin
import numpy as np
from visualization_2 import VizScene
import transforms as tr
from scipy.io import loadmat
import sympy as sp

# -----------------------------
# Define 6-DOF arm (full 3 position + 3 orientation control)
# -----------------------------
# 6-DOF provides complete control over end effector position and orientation
# Designed for pen-writing: joints separated with proper offsets, no overlaps
dh_params = [
    [0, 0.5, 0.0, np.pi/2.0],   # Joint 1: base rotation (0.5m vertical offset)
    [0, 0.0, 0.8, 0],           # Joint 2: shoulder (0.8m horizontal link)
    [0, 0.0, 0.8, 0],           # Joint 3: elbow (0.8m horizontal link)  
    [0, 0.3, 0.0, np.pi/2.0],   # Joint 4: wrist roll (0.3m offset for clearance)
    [0, 0.0, 0.0, -np.pi/2.0],  # Joint 5: wrist pitch
    [0, 0.2, 0.0, 0]            # Joint 6: wrist yaw (0.2m to pen mount)
]

# Tip frame: rotate so the visualization arrow (X-axis) points down
# The red arrow in the visualization is the X-axis of the end effector
# Rotate -90° around Y to make X point down (was pointing forward)
# Sphere (end effector) now attaches directly to the last joint
tip_rot = tr.roty(-np.pi/2)
tip = tr.se3(tip_rot, np.array([0, 0, 0]))

arm = kin.SerialArm(dh_params, jt=['r']*6, tip=tip)

# Desired poses - X-axis pointing down (for visualization arrow)
# X-axis = down, Y-axis = right, Z-axis = forward (right-handed frame)
x_des = np.array([0, 0, -1])  # X points down (red arrow)
y_des = np.array([0, 1, 0])   # Y points right (green arrow)
z_des = np.array([1, 0, 0])   # Z points forward (blue arrow)
R_des = np.column_stack([x_des, y_des, z_des])

# Read signature trajectory from CSV 
signature_trajectory = loadmat('P_TylerWaite.mat')
path = signature_trajectory['path']  # Extract path data
print('The numba')
sp.pprint(path[:,1])
print('The end of numba')


# Signature trajectory: cursive "Davis" with pen lifts
# Paper surface at Z=0.0m (table), pen lifts to Z=0.05m (5cm above paper)
# Format: (position, orientation, pen_down)
# pen_down=True: pen touching paper, pen_down=False: pen lifted
# signature_trajectory = [
#     # Letter "D" - start at left
#     (np.array([1.1, -0.25, 0.05]), R_des, False),  # Move to start (pen up)
#     (np.array([1.1, -0.25, 0.0]), R_des, True),    # Lower pen
#     (np.array([1.1, -0.25, 0.00]), R_des, True),   # Draw vertical line up
#     (np.array([1.15, -0.23, 0.00]), R_des, True),  # Curve top right
#     (np.array([1.18, -0.20, 0.00]), R_des, True),  # Continue curve
#     (np.array([1.18, -0.20, 0.0]), R_des, True),   # Curve back down
#     (np.array([1.15, -0.23, 0.0]), R_des, True),   # Close the D
#     (np.array([1.1, -0.25, 0.0]), R_des, True),    # Back to start
    
#     # Pen lift and move to "a"
#     (np.array([1.18, -0.20, 0.05]), R_des, False), # Lift pen
#     (np.array([1.2, -0.15, 0.00]), R_des, False),  # Move to next letter
    
#     # Letter "a" - small circle with tail
#     (np.array([1.2, -0.15, 0.05]), R_des, True),   # Lower pen (mid height)
#     (np.array([1.18, -0.13, 0.00]), R_des, True),  # Small circle top
#     (np.array([1.18, -0.13, 0.0]), R_des, True),   # Circle right side
#     (np.array([1.2, -0.15, 0.0]), R_des, True),    # Circle bottom
#     (np.array([1.22, -0.13, 0.00]), R_des, True),  # Right tail up
    
#     # Letter "v" - continuous from "a"
#     (np.array([1.25, -0.10, 0.05]), R_des, True),   # Left stroke down
#     (np.array([1.27, -0.08, 0.00]), R_des, True),  # Right stroke up
    
#     # Letter "i" - dot will be separate
#     (np.array([1.30, -0.05, 0.05]), R_des, True),   # Vertical stroke down
#     (np.array([1.30, -0.05, 0.00]), R_des, True),  # Stroke up
    
#     # Pen lift for dot
#     (np.array([1.30, -0.05, 0.05]), R_des, False), # Lift for dot
#     (np.array([1.30, -0.05, 0.00]), R_des, True),  # Dot (brief touch)
#     (np.array([1.30, -0.05, 0.00]), R_des, False), # Lift again
    
#     # Move to "s"
#     (np.array([1.32, -0.02, 0.05]), R_des, False), # Move to S
#     (np.array([1.32, -0.02, 0.00]), R_des, True),  # Lower pen
    
#     # Letter "s" - snake curve
#     (np.array([1.33, 0.0, 0.05]), R_des, True),    # Upper curve right
#     (np.array([1.34, 0.0, 0.00]), R_des, True),    # Curve back
#     (np.array([1.35, 0.02, 0.00]), R_des, True),    # Lower curve left
#     (np.array([1.36, 0.02, 0.00]), R_des, True),    # End stroke
    
#     # Final pen lift
#     (np.array([1.36, 0.02, 0.05]), R_des, False),  # Lift pen off paper
# ]

# Extract just the poses for the main loop (will handle pen state separately)
# desired_poses = [(pos, rot) for pos, rot, _ in signature_trajectory]
desired_poses = path
# pen_states = [pen_down for _, _, pen_down in signature_trajectory]
# Initial configuration for 6-DOF
q_0 = np.array([0.0, np.pi/6, np.pi/6, 0.0, -np.pi/3, 0.0])

# Visualization
viz = VizScene()
viz.add_arm(arm)
viz.update(qs=[q_0])

print("="*60)
print("SIGNATURE WRITING - 6-DOF ARM WITH PEN CONTROL")
print("="*60)
print(f"Total waypoints: {len(desired_poses)}")
print("Pen states: UP (not drawing) | DOWN (drawing)")
print("="*60)

# -----------------------------
# Two-Phase Control Strategy with Pen State Tracking
# -----------------------------
for pose_idx in range(np.shape(path)[1]):
    print(len(path))
    p_des = path[:,pose_idx]
    p_des[0] = p_des[0] + 1  # Scale X for better reach
    p_des[1] = p_des[1]   # Scale Y for better centering
    p_des[2] = p_des[2] *.5  # Scale Z for pen height (0 or 0.05m)
    pen_down = p_des[2]
    pen_status = "DOWN ✓ (drawing)" if pen_down else "UP   (not drawing)"
    
    print(f"\n>>> Waypoint {pose_idx+1}/{len(desired_poses)}: Target = {p_des}")
    print(f"    Pen state: {pen_status}")
    print(f"    Target X-axis = [0, 0, -1] (red arrow pointing down)")
    
    # PHASE 1: Achieve position (relaxed orientation)
    print("\n  Phase 1: Moving to position...")
    prev_pos_err = float('inf')
    stuck_count = 0
    
    for iter in range(300):
        T_cur = arm.fk(q_0, tip=True)
        p_cur = T_cur[:3,3]
        R_cur = T_cur[:3,:3]
        
        pos_err = p_des - p_cur
        pos_err_mag = np.linalg.norm(pos_err)
        
        # Check if making progress
        if abs(pos_err_mag - prev_pos_err) < 1e-4:
            stuck_count += 1
            if stuck_count > 20:
                print(f"    Position not improving, stopping Phase 1 at error={pos_err_mag:.4f}")
                break
        else:
            stuck_count = 0
        prev_pos_err = pos_err_mag
        
        # Soft orientation constraint (just keep X somewhat downward)
        x_cur = R_cur[:, 0]  # X-axis (red arrow)
        x_des_vec = R_des[:, 0]
        ori_err = np.cross(x_cur, x_des_vec)  # Small when aligned
        
        # Combine errors with high position priority
        error = np.hstack((pos_err, 0.2 * ori_err))  # Scale down orientation
        
        if iter % 50 == 0 or iter == 0:
            x_dot = np.dot(x_cur, x_des_vec)
            print(f"    Iter {iter:3d}: pos_err={pos_err_mag:.4f}, X-align={x_dot:.3f}")
        
        # Tighter tolerance for writing (2mm accuracy)
        if pos_err_mag < 0.002:
            print(f"    Position reached in {iter} iterations!")
            break
        
        # Update with Jacobian - adaptive step size
        J = arm.jacob(q_0, tip=True)
        kd = 0.1  # Increased damping for stability
        JJT = J @ J.T
        dq = J.T @ np.linalg.inv(JJT + kd**2 * np.eye(6)) @ error
        
        # Smooth motion for writing - smaller steps
        step_size = 0.15 if pos_err_mag > 0.3 else 0.25
        q_0 = q_0 + step_size * dq
        q_0 = np.clip(q_0, -np.pi, np.pi)
        viz.update(qs=[q_0])
        viz.hold(0.015)  # Smooth visualization for writing
    
    # PHASE 2: Refine orientation (maintain position)
    print("\n  Phase 2: Refining Z-axis orientation...")
    for iter in range(300):
        T_cur = arm.fk(q_0, tip=True)
        p_cur = T_cur[:3,3]
        R_cur = T_cur[:3,:3]
        
        pos_err = p_des - p_cur
        
        # Full orientation error
        R_err = R_des @ R_cur.T
        axis_angle = tr.R2axis(R_err)
        angle, axis = axis_angle[0], axis_angle[1:]
        ori_err = angle * axis if np.abs(angle) > 1e-6 else np.zeros(3)
        
        # Balance position and orientation
        error = np.hstack((pos_err, 2.0 * ori_err))  # Higher orientation weight
        
        x_cur = R_cur[:, 0]  # X-axis (red arrow)
        x_des_vec = R_des[:, 0]
        x_align = np.dot(x_cur, x_des_vec)
        
        if iter % 50 == 0 or iter == 0:
            print(f"    Iter {iter:3d}: pos_err={np.linalg.norm(pos_err):.4f}, ori_err={np.linalg.norm(ori_err):.4f}, X-align={x_align:.3f}")
        
        # Check if converged
        if np.linalg.norm(pos_err) < .01 and abs(1.0 - x_align) < .01:
            print(f"    [SUCCESS] Full convergence in {iter} iterations!")
            print(f"    Final X-alignment: {x_align:.6f}")
            break
        
        if iter == 299:
            print(f"    [PARTIAL] Position close, X-alignment: {x_align:.3f}")
        
        # Update with Jacobian
        J = arm.jacob(q_0, tip=True)
        kd = 0.08
        JJT = J @ J.T
        dq = J.T @ np.linalg.inv(JJT + kd**2 * np.eye(6)) @ error
        q_0 = q_0 + 0.025 * dq
        q_0 = np.clip(q_0, -np.pi, np.pi)
        viz.update(qs=[q_0])
        viz.hold(0.015)  # Smooth visualization for writing

    # viz.hold() # Pause to look at robot
    # Summary with pen state
    T_final = arm.fk(q_0, tip=True)
    p_final = T_final[:3,3]
    x_final = T_final[:3,0]  # Red arrow (X-axis)
    x_align_final = np.dot(x_final, [0,0,-1])
    
    drawing_status = "DRAWING ✓" if pen_down else "MOVING (pen up)"
    print(f"\n  ✓ Waypoint complete - {drawing_status}")
    print(f"    Position error: {np.linalg.norm(p_des - p_final):.6f} m")
    print(f"    X-alignment: {x_align_final:.4f}")

print("\n" + "="*60)
print("SIGNATURE COMPLETE - Visualization window remains open")
print(f"Total waypoints: {len(path[0,:])}")
print(f"Drawing waypoints: {sum(path[2,:])}")
print(f"Movement waypoints: {len(path[2,:]) - sum(path[2,:])}")
print("="*60)
