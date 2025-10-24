# %% [markdown]
# # Midterm 2025
# * Copy this file to your homework workspace to have access to your other kinematic and visualization functions

# %%
# To test your setup, after defining the robot arm as described below, (but nothing else)
# you can run this file directly to make sure it is plotting the arm, obstacle, and goal 
# as expected. 

import kinematics as kin  #this is your kinematics file that you've been developing all along
from visualization import VizScene #this is the visualization file you've been using for homework
import time
import numpy as np


# Define your kinematics and an "arm" variable here using DH parameters so they
# are global variables that are available in your function below:

dh = np.array([[np.pi/2, 2, 0, np.pi/2], 
               [np.pi/2-np.pi/3, 0, 0, np.pi/2], 
               [0, 4, 0, np.pi/2], 
               [np.pi/2-np.pi/3, 0, 2, np.pi/2]])

#First joint is up in the air, so this will place it where it is needed
base = np.array([[1,0,0,0], 
                 [0,1,0,0],
                 [0,0,1,2],
                 [0,0,0,1]])
jt_types = np.array(['r', 'r', 'r', 'r'])
arm = kin.SerialArm(dh,jt=jt_types, base=base)

#Define reach for error checking
reach = 0
for dh in dh:
      reach += np.linalg.norm(dh[1:3])


# let's also plot robot to make sure it matches what we think it should
# (this will look mostly like the pictures on part 1 if your DH parameters
# are correct)
viz_check = VizScene()
viz_check.add_arm(arm, joint_colors=[np.array([0.95, 0.13, 0.13, 1])]*arm.n)
viz_check.update(qs = [[0, 0, 0, 0]])
viz_check.hold()


def compute_robot_path(q_init, goal, obst_location, obst_radius):
      #Error checking
      if isinstance(q_init, np.ndarray):
            q = q_init
      elif isinstance(q_init, list):
            q = np.array(q_init)
      else:
            raise TypeError("Invlid type for initial joint positions 'q0'")

      #List of angles between q_init and the goal
      q_s = []

      #define variables needed
      cur_position = arm.fk(q, base=base)[0:3,3]
      error = goal - cur_position
      p_b = 0
      p_0 = obst_radius * 1.8 #Max range of repulsive field
      xi = 1.1  #attractive potential gain
      k =  10  #repulsive potential gains
      damping_coefficient = 1.4   # to damp the pseudo inverse Jacobian
      max_iter = 1000              # how many iterations to run before stopping
      tol = .000001                 
      step_scale = .05


      iters = 0
      while np.linalg.norm(error) > tol and iters < max_iter:
      #     TODO fill this out with your path planning method

            # Position information used to determine the attraction and repulsion forces, and thus velocities
            cur_position = arm.fk(q, base=base)[0:3,3]
            cur_x = cur_position[0]
            cur_y = cur_position[1]
            cur_z = cur_position[2]

            jt4_position = arm.fk(q,index=[0,4])[0:3,3]
            jt4_x = jt4_position[0]
            jt4_y = jt4_position[1]
            jt4_z = jt4_position[2]

            obstacle_x = obst_location[0]
            obstacle_y = obst_location[1]
            obstacle_z = obst_location[2]

            J = arm.jacob(q)[0:3, :]
            J_jt4 = arm.jacob(q, index=3)[0:3, :]
            #Attraction and repulsion field variables
            error = goal - cur_position  #distance between end effector and target
            p_b_end_effector = np.sqrt((cur_x-obstacle_x)**2+(cur_y-obstacle_y)**2+(cur_z-obstacle_z)**2)-obst_radius  #minimum distance between the robot's end effector and the obstacle
            p_b_jt4 = np.sqrt((jt4_x-obstacle_x)**2+(jt4_y-obstacle_y)**2+(jt4_z-obstacle_z)**2)-obst_radius          #minimum distance between the robot's joint 4 and the obstacle

            J_damped_pseudo = J.T@ np.linalg.inv(J @ J.T + damping_coefficient**2 * np.eye(J.shape[0]))       # Damped Pseduo inverse of J used to find velocities
            J_damped_pseudo_jt4 = J_jt4.T @ np.linalg.inv(J_jt4 @ J_jt4.T + damping_coefficient**2 * np.eye(J_jt4.shape[0])) 
            
            #Attraction Force
            vatt = xi * error
            qdelta_attraction = J_damped_pseudo @ vatt


            #Find replusive forces for both the end effector and joint 4
            if p_b_end_effector <= p_0:
                  direction_of_repulsion_end_effector = (cur_position - obst_location)/p_b_end_effector #unit vector pointing away from obstacle for end effector
                  vrep_end_effector = k*((1/p_b_end_effector)-(1/p_0))*(1/(p_b_end_effector**2)) * direction_of_repulsion_end_effector
            else:
                  vrep_end_effector = np.array([0.,0.,0.])

            if p_b_jt4 <= p_0:
                  direction_of_repulsion_jt4 = (jt4_position - obst_location)/p_b_jt4 #unit vector pointing away from obstacle for end effector
                  vrep_jt4 = k*((1/p_b_jt4)-(1/p_0))*(1/(p_b_jt4**2)) * direction_of_repulsion_jt4
            else:
                  vrep_jt4 = np.array([0.,0.,0.])
            
            #Find combined repulsion delta q for both the end effector and joint 4
            qdelta_repulsion = (J_damped_pseudo @ vrep_end_effector) + (J_damped_pseudo_jt4 @ vrep_jt4)


            #Calculate qdelta resulting from both the attraction and the repulsion
            qdelta = qdelta_attraction + qdelta_repulsion

            #Add a feature so that if the arm gets stuck on a "minimum" it will begin rotating the first joint away from the minimum
            if np.linalg.norm(qdelta) < 0.1 and iters < 300:
                  qdelta += np.array([.1, 0, 0, 0])

            #Find the next q
            q =  q + step_scale * qdelta

            #Calculate the new current position and error for next cycle
            cur_position = arm.fk(q, base=base)[0:3,3]
            error = goal - cur_position  #distance between end effector and target

            q_s.append(q)
            iters += 1
      return q_s


if __name__ == "__main__":

      # if your function works, this code should show the goal, the obstacle, and your robot moving towards the goal.
      # Please remember that to test your function, I will change the values below to see if the algorithm still works.
      q_0 = [0, 0, 0, 0]
      goal = [0,2,4]
      obst_position = [0, 3, 2]
      obst_rad = 1.0

      q_ik_slns = compute_robot_path(q_0, goal, obst_position, obst_rad)


      # if you just want to check if you have your code set up and arm defined correctly, you can uncomment the next three lines 
      # and run this file using either vs code or the terminal (and running "python3 midterm_2025.py"). None of the next three 
      # lines are needed for your solution, they may just help you check your visualization before you get going. It will just 
      # display 100 different random sets of joint angles as well as the goal and obstacle.

      # import numpy as np
      # q_ik_slns = np.random.uniform(size=(100,4))
      # q_ik_slns = q_ik_slns.tolist()


      # depending on how you store q_ik_slns inside your function, you may need to change this for loop
      # definition. However if you store q as I've done above, this should work directly.
      viz = VizScene()
      viz.add_arm(arm, joint_colors=[np.array([0.95, 0.13, 0.13, 1])]*arm.n)
      viz.add_marker(goal, radius=0.1)
      viz.add_obstacle(obst_position, rad=obst_rad)
      for q in q_ik_slns:
            viz.update(qs=[q])

            # if your step in q is very small, you can shrink this time, or remove it completely to speed up your animation
            time.sleep(0.05)
