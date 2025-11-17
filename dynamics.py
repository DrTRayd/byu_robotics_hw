"""
dynamics Module - Contains code for:
- Dynamic SerialArm class
- RNE Algorithm
- Euler - Lagrange formulation

John Morrell, Jan 28 2022
Tarnarmour@gmail.com

modified by:
Marc Killpack, Nov. 4, 2022
"""

import numpy as np
from kinematics import SerialArm
from utility import skew

eye = np.eye(4)

class SerialArmDyn(SerialArm):
    """
    SerialArmDyn class represents serial arms with dynamic properties and is used to calculate forces, torques, accelerations,
    joint forces, etc. using the Newton-Euler and Euler-Lagrange formulations. It inherits from the previously defined kinematic
    robot arm class "SerialArm".
    """

    def __init__(self,
                 dh,
                 jt=None,
                 base=eye,
                 tip=eye,
                 joint_limits=None,
                 mass=None,
                 r_com=None,
                 link_inertia=None,
                 motor_inertia=None,
                 joint_damping=None):

        SerialArm.__init__(self, dh, jt, base, tip, joint_limits)
        self.mass = mass
        self.r_com = r_com
        self.link_inertia = link_inertia
        self.motor_inertia = motor_inertia
        if joint_damping is None:
            self.B = np.zeros((self.n, self.n))
        else:
            self.B = np.diag(joint_damping)

    def rne(self, q, qd, qdd,
            Wext=np.zeros((6,1)),
            g=np.zeros((3, 1)),
            omega_base=np.zeros((3, 1)),
            alpha_base=np.zeros((3, 1)),
            v_base=np.zeros((3, 1)),
            acc_base=np.zeros((3, 1))):

        """
        tau, W = RNE(q, qd, qdd):
        returns the torque in each joint (and the full wrench at each joint) given the joint configuration, velocity, and accelerations
        Args:
            q:
            qd:
            qdd:

        Returns:
            tau: torques or forces at joints (assuming revolute joints for now though)
            wrenches: force and torque at each joint, and for joint i, the wrench is in frame i


        We start with the velocity and acceleration of the base frame, v0 and a0, and the joint positions, joint velocities,
        and joint accelerations (q, qd, qdd).

        For each joint, we find the new angular velocity, w_i = w_(i-1) + z * qdot_(i-1)
        v_i = v_(i-1) + w_i x r_(i-1, com_i)


        if motor inertia is None, we don't consider it. Solve for now without motor inertia. The solution will provide code for motor inertia as well.
        """

        # Normalize vector inputs to 1D arrays to avoid unexpected broadcasting
        Wext = np.asarray(Wext).reshape(-1)
        g = np.asarray(g).reshape(-1)
        omega_base = np.asarray(omega_base).reshape(-1)
        alpha_base = np.asarray(alpha_base).reshape(-1)
        v_base = np.asarray(v_base).reshape(-1)
        acc_base = np.asarray(acc_base).reshape(-1)

        omegas = []
        alphas = []
        v_ends = []
        v_coms = []
        acc_ends = []
        acc_coms = []

        ## Make space for n+1 elements (including base)
        for i in range(self.n + 1):
            if i == 0:
                omegas.append(omega_base)
                alphas.append(alpha_base)
                v_ends.append(v_base)
                v_coms.append(v_base)
                acc_ends.append(acc_base)
                acc_coms.append(acc_base)
            else:
                omegas.append(np.zeros(3))
                alphas.append(np.zeros(3))
                v_ends.append(np.zeros(3))
                v_coms.append(np.zeros(3))
                acc_ends.append(np.zeros(3))
                acc_coms.append(np.zeros(3))

        # Empty rotation matrix
        R = [0]* (self.n + 1)
        p = [0]* (self.n + 1)

        ## Solve for needed angular velocities, angular accelerations, and linear accelerations
        ## If helpful, you can define a function to call here so that you can debug the output more easily.
        for i in range(0, self.n):
            
            # Get the dh parameters for joint i
            a_i, alpha_i, d_i, theta_i = self.dh[i]

            #Compute the actual theta and d based on joint type
            if self.jt[i] == 'r':
                theta = theta_i + q[i]
                d = d_i
            else:
                theta = theta_i
                d = d_i + q[i]

            #Find rotation matrix from i-1 to i
            cos_theta = np.cos(theta)
            sin_theta = np.sin(theta)
            cos_alpha = np.cos(alpha_i)
            sin_alpha = np.sin(alpha_i)

            R[i+1] = np.array([[cos_theta, -sin_theta * cos_alpha, sin_theta * sin_alpha],
                            [sin_theta, cos_theta * cos_alpha, -cos_theta * sin_alpha],
                            [0, sin_alpha, cos_alpha]])
            
            #Find position matrix from i-1 to i
            p_i = np.array([a_i*cos_theta, a_i*sin_alpha, d])
            # store position vector for use in backward pass
            p[i+1] = p_i

            #Find the z axis for joint i-1
            z_i_1 = np.array([0,0,1])  # third column of rotation matrix

            #Solve for angular velocity
            omegas[i+1] = R[i+1].T @ omegas[i] + z_i_1 * qd[i]

            #Angular acceleration
            alphas[i+1] = R[i+1].T @ alphas[i] + np.cross(R[i+1].T @ omegas[i], z_i_1 * qd[i]) + z_i_1 * qdd[i]

            #Linear velocity of origin of frame i
            v_ends[i+1] = R[i+1].T @ (v_ends[i] + np.cross(omegas[i], p_i))

            #Linear acceleration of origin of frame i
            acc_ends[i+1] = R[i+1].T @ (acc_ends[i] + np.cross(alphas[i], p_i) + np.cross(omegas[i], np.cross(omegas[i], p_i)))

            #Linear velocity of center of mass of link i
            r_com_i = self.r_com[i]
            v_coms[i] = v_ends[i+1] + np.cross(omegas[i+1], r_com_i)

            #Linear acceleration of center of mass of link i
            acc_coms[i] = acc_ends[i+1] + np.cross(alphas[i+1], r_com_i) + np.cross(omegas[i+1], np.cross(omegas[i+1], r_com_i))

        ## Now solve Kinetic equations by starting with forces at last link and going backwards
        ## If helpful, you can define a function to call here so that you can debug the output more easily.
        # Use independent 6-element 1D arrays for wrenches (force then moment)
        Wrenches = [np.zeros(6) for _ in range(self.n + 1)]
        tau = [0] * self.n

        for i in range(self.n - 1, -1, -1):  # Index from n-1 to 0
            I = self.link_inertia[i]

            F_i = self.mass[i] * acc_coms[i]
            N_i = I @ alphas[i+1] + np.cross(omegas[i+1], I @ omegas[i+1])

            # Transform forces and moments from link i+1 to link i
            R_next = R[i+1]
            p_next = p[i+1]

            #Transform wrench from frame i+1 to frame i
            F_child = R_next @ Wrenches[i+1][0:3]
            N_child = R_next @ Wrenches[i+1][3:6] + np.cross(p_next, F_child)
            
            # Total force and moment at joint i
            F_total = F_i + F_child
            N_total = N_i + N_child + np.cross(self.r_com[i], F_i) + np.cross(p_next, F_child)
            Wrenches[i] = np.hstack((F_total, N_total))

            # Project wrench onto joint axis to find torque/force
            z_i = R[i+1][:, 2]  # joint axis in frame i
            tau[i] = z_i @ N_total


        print("Joint Torques/Forces:")
        print(tau)
        print("Wrenches:")
        for i in range(len(Wrenches)):
            print(f"Joint {i} Wrench: {Wrenches[i]}")
        return tau, Wrenches
    

if __name__ == '__main__':

    ## NOTE: These are not the parameters for the HW. This is just trying to show
    ## an example of how to use the code.

    ## this just gives an example of how to define a robot, this is a planar 3R robot.
    dh = [[0, 0, 1, 0],
          [0, 0, 1, 0],
          [0, 0, 1, 0]]

    joint_type = ['r', 'r', 'r']

    link_masses = [1, 1, 1]

    # defining three different centers of mass, one for each link
    r_coms = [np.array([-0.5, 0, 0]), np.array([-0.5, 0, 0]), np.array([-0.5, 0, 0])]

    link_inertias = []
    for i in range(len(joint_type)):
        iner = link_masses[i] / 12 * dh[i][2]**2

        # this inertia tensor is only defined as having Iyy, and Izz non-zero
        link_inertias.append(np.array([[0, 0, 0], [0, iner, 0], [0, 0, iner]]))


    arm = SerialArmDyn(dh,
                       jt=joint_type,
                       mass=link_masses,
                       r_com=r_coms,
                       link_inertia=link_inertias)

    # once implemented, you can call arm.RNE and it should work.
    q = [np.pi/4.0]*3
    qd = [0.2]*3
    qdd = [0.05]*3
    arm.rne(q, qd, qdd)
