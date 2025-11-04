"""
Kinematics Module - Contains code for:
- Forward Kinematics, from a set of DH parameters to a serial linkage arm with callable forward kinematics
- Inverse Kinematics
- Jacobian

John Morrell, Jan 26 2022
Tarnarmour@gmail.com

modified by: 
Marc Killpack, Sept 21, 2022 and Sept 21, 2023
"""


from transforms import *

eye = np.eye(4)
pi = np.pi

# this is a convenience class that makes it easy to define a function that calculates "A_i(q)", given the
# DH parameters for link and joint "i" only. 
class dh2AFunc:
    """
    A = dh2AFunc(dh, joint_type="r")
    Description:
    Accepts a list of 4 dh parameters corresponding to the transformation for one link 
    and returns a function "f" that will generate a homogeneous transform "A" given 
    "q" as an input. A represents the transform from link i-1 to link i. This follows
    the "standard" DH convention. 

    Parameters:
    dh - 1 x 4 list from dh parameter table for one transform from link i-1 to link i,
    in the order [theta d a alpha] - THIS IS NOT THE CONVENTION IN THE BOOK!!! But it is the order of operations. 

    Returns:
    f(q) - a function that can be used to generate a 4x4 numpy matrix representing the homogeneous transform 
        from one link to the next
    """
    def __init__(self, dh, jt):

        # if joint is revolute implement correct equations here:
        if jt == 'r':
            # although A(q) is only a function of "q", the dh parameters are available to these next functions 
            # because they are passed into the "init" function above. 

            def A(q):
                # See eq. (2.52), pg. 64
                # TODO - complete code that defines the "A" or "T" homogenous matrix for a given set of DH parameters. 
                # Do this in terms of the variables "dh" and "q" (so that one of the entries in your dh list or array
                # will need to be added to q).
                
                # start by assigning the DH parameters to be readable
                # notice that in the case of a revolute joint, q is added to the fixed "theta" offset (if there is one)
                theta = dh[0] + q
                d = dh[1]
                a = dh[2]
                alpha = dh[3]

                cth = np.cos(theta)
                sth = np.sin(theta)
                cal = np.cos(alpha)
                sal = np.sin(alpha)

                return np.array(
                    [[cth, -sth * cal, sth *sal, a * cth],
                     [sth, cth * cal, -cth * sal, a * sth],
                     [0, sal, cal, d],
                     [0, 0, 0, 1]])


        # if joint is prismatic implement correct equations here:
        else:
            def A(q):
                # See eq. (2.52), pg. 64
                # TODO - complete code that defines the "A" or "T" homogenous matrix for a given set of DH parameters. 
                # Do this in terms of the variables "dh" and "q" (so that one of the entries in your dh list or array
                # will need to be added to q).
                
                # notices that in the case of a prismatic joint, q is added to the fixed "d" parameter
                theta = dh[0]
                d = dh[1] + q
                a = dh[2]
                alpha = dh[3]

                cth = np.cos(theta)
                sth = np.sin(theta)
                cal = np.cos(alpha)
                sal = np.sin(alpha)

                return np.array(
                    [[cth, -sth * cal, sth * sal, a * cth],
                     [sth, cth * cal, -cth * sal, a * sth],
                     [0, sal, cal, d],
                     [0, 0, 0, 1]])


        self.A = A


class SerialArm:
    """
        SerialArm - A class designed to represent a serial link robot arm

        SerialArms have frames 0 to n defined, with frame 0 located at the first joint and aligned with the robot body
        frame, and frame n located at the end of link n.
    """


    def __init__(self, dh, jt=None, base=eye, tip=eye, joint_limits=None):
        """
            arm = SerialArm(dh_params, joint_type, base=I, tip=I, radians=True, joint_limits=None)
            :param dh: n length list where each entry in list is another list of length 4, representing dh parameters, [theta d a alpha]
            :param jt: n length list of strings, 'r' for revolute joint and 'p' for prismatic joint
            :param base: 4x4 numpy array representing SE3 transform from world or inertial frame to frame 0
            :param tip: 4x4 numpy array representing SE3 transform from frame n to tool frame or tip of robot
            :param joint_limits: 2 length list of n length lists, holding first negative joint limit then positive, none for
            not implemented
        """
        self.dh = dh
        self.n = len(dh)

        # we will use this list to store the A matrices for each set/row of DH parameters. 
        self.transforms = []

        # assigning a joint type
        if jt is None:
            self.jt = ['r'] * self.n
        else:
            self.jt = jt
            if len(self.jt) != self.n:
                print("WARNING! Joint Type list does not have the same size as dh param list!")
                return None

        # using the code we wrote above to generate the function A(q) for each set of DH parameters
        for i in range(self.n):
            # TODO use the class definition above (dh2AFunc), and the dh parameters and joint type to
            # make a function and then append that function to the "transforms" list. 
            f = dh2AFunc(dh[i], self.jt[i])
            self.transforms.append(f.A)

        # assigning the base, and tip transforms that will be added to the default DH transformations.
        self.base = base
        self.tip = tip
        self.qlim = joint_limits

        # Calculate the reach of the robot arm
        self.reach = sum(dh[i][2] for i in range(self.n))  # Sum of the 'a' parameters

    def fk(self, q, index=None, base=False, tip=False):
        """
            T = arm.fk(q, index=None, base=False, tip=False)
            Description: 
                Returns the transform from a specified frame to another given a 
                set of joint inputs q and the index of joints

            Parameters:
                :param q - list or iterable of floats which represent the joint positions
                :param index - integer or list of two integers. If a list of two integers, the first integer represents the starting JOINT 
                    (with 0 as the first joint and n as the last joint) and the second integer represents the ending FRAME
                    If one integer is given only, then the integer represents the ending Frame and the FK is calculated as starting from 
                    the first joint
                :param base - bool, if True then if index starts from 0 the base transform will also be included
                :param tip - bool, if true and if the index ends at the nth frame then the tool transform will be included
            
            Returns:
                T - the 4 x 4 homogeneous transform from frames determined from "index" variable
        """

        ###############################################################################################
        # the following lines of code are data type and error checking. You don't need to understand
        # all of it, but it is helpful to keep. 

        if not hasattr(q, '__getitem__'):
            q = [q]

        if len(q) != self.n:
            print("WARNING: q (input angle) not the same size as number of links!")
            return None

        if isinstance(index, (list, tuple)):
            start_frame = index[0]
            end_frame = index[1]
        elif index == None:
            start_frame = 0
            end_frame = self.n
        else:
            start_frame = 0
            if index < 0:
                print("WARNING: Index less than 0!")
                print(f"Index: {index}")
                return None
            end_frame = index

        if end_frame > self.n:
            print("WARNING: Ending index greater than number of joints!")
            print(f"Starting frame: {start_frame}  Ending frame: {end_frame}")
            return None
        if start_frame < 0:
            print("WARNING: Starting index less than 0!")
            print(f"Starting frame: {start_frame}  Ending frame: {end_frame}")
            return None
        if start_frame > end_frame:
            print("WARNING: starting frame must be less than ending frame!")
            print(f"Starting frame: {start_frame}  Ending frame: {end_frame}")
            return None
        ###############################################################################################
        ###############################################################################################         

        # TODO - Write code to calculate the total homogeneous transform "T" based on variables stored
        # in "base", "tip", "start_frame", and "end_frame". Look at the function definition if you are 
        # unsure about the role of each of these variables. This is mostly easily done with some if/else 
        # statements and a "for" loop to add the effect of each subsequent A_i(q_i). But you can 
        # organize the code any way you like.  
        
        if base and start_frame == 0:
            T = self.base
        else:
            T = eye

        for i in range(start_frame, end_frame):
            T = T @ self.transforms[i](q[i])

        if tip and end_frame == self.n:
            T = T @ self.tip

        return T


    def __str__(self):
        """
            This function just provides a nice interface for printing information about the arm. 
            If we call "print(arm)" on an SerialArm object "arm", then this function gets called.
            See example in "main" below. 
        """
        dh_string = """DH PARAMS\n"""
        dh_string += """theta\t|\td\t|\ta\t|\talpha\t|\ttype\n"""
        dh_string += """---------------------------------------\n"""
        for i in range(self.n):
            dh_string += f"{self.dh[i][0]}\t|\t{self.dh[i][1]}\t|\t{self.dh[i][2]}\t|\t{self.dh[i][3]}\t|\t{self.jt[i]}\n"
        return "Serial Arm\n" + dh_string
    



    def jacob(self, q: list[float]|NDArray, index: int|None=None, base: bool=False,
              tip: bool=False) -> NDArray:
        """
        J = arm.jacob(q)

        Calculates the geometric jacobian for a specified frame of the arm in a given configuration

        :param list[float] | NDArray q: joint positions
        :param int | None index: joint frame at which to calculate the Jacobian
        :param bool base: specify whether to include the base transform in the Jacobian calculation
        :param bool tip: specify whether to include the tip transform in the Jacobian calculation
        :return J: 6xN numpy array, geometric jacobian of the robot arm
        """

        if index is None:
            index = self.n
        assert 0 <= index <= self.n, 'Invalid index value!'

        # TODO - start by declaring a zero matrix that is the correct size for the Jacobian
        J = np.zeros((6, len(q)))

        # TODO - find the current position of the point of interest (usually origin of frame "n")
        # using your fk function this will likely require additional intermediate variables than
        # what is shown here.
        pe = self.fk(q, index=index, base=base, tip=tip)[:3, 3]


        # TODO - calculate all the necessary values using your "fk" function, and fill every column
        # of the jacobian using this "for" loop. Functions like "np.cross" may also be useful.
        for i in range(index):
            # check if joint is revolute
            if self.jt[i] == 'r':
                # find z axis of joint i
                Ti = self.fk(q, index=[0, i], base=base, tip=False)
                zi = Ti[:3, 2]

                # find position of joint i
                pi = Ti[:3, 3]

                # calculate linear velocity component
                Jv = np.cross(zi, (pe - pi))

                # calculate angular velocity component
                Jw = zi

                # fill in the Jacobian matrix
                J[:, i] = np.hstack((Jv, Jw))

            # if not assume joint is prismatic
            else:
                # find z axis of joint i
                Ti = self.fk(q, index=[0, i], base=base, tip=False)
                zi = Ti[:3, 2]

                # calculate linear velocity component
                Jv = zi

                # calculate angular velocity component
                Jw = np.zeros(3)

                # fill in the Jacobian matrix
                J[:, i] = np.hstack((Jv, Jw))


        return J
    
    def ik_position(self, target: NDArray, q0: list[float]|NDArray|None=None,
                    method: str='J_T', force: bool=True, tol: float=1e-4,
                    K: NDArray=None, kd: float=0.001, max_iter: int=100,
                    debug: bool=False, debug_step: bool=False
                    ) -> tuple[NDArray, NDArray, int, bool]:
        """
        qf, error_f, iters, converged = arm.ik_position(target, q0, 'J_T', K=np.eye(3))

        Computes the inverse kinematics solution (position only) for a given target
        position using a specified method by finding a set of joint angles that
        place the end effector at the target position without regard to orientation.

        :param NDArray target: 3x1 numpy array that defines the target location.
        :param list[float] | NDArray | None q0: list or array of initial joint positions,
            defaults to q0=0 (which is often a singularity - other starting positions
            are recommended).
        :param str method: select which IK algorithm to use. Options include:
            - 'pinv': damped pseudo-inverse solution, qdot = J_dag * e * dt, where
            J_dag = J.T * (J * J.T + kd**2)^-1
            - 'J_T': jacobian transpose method, qdot = J.T * K * e
        :param bool force: specify whether to attempt to solve even if a naive reach
            check shows the target is outside the reach of the arm.
        :param float tol: tolerance in the norm of the error in pose used as
            termination criteria for while loop.
        :param NDArray K: 3x3 numpy array. For both pinv and J_T, K is the positive
            definite gain matrix.
        :param float kd: used in the pinv method to make sure the matrix is invertible.
        :param int max_iter: maximum attempts before giving up.
        :param bool debug: specify whether to plot the intermediate steps of the algorithm.
        :param bool debug_step: specify whether to pause between each iteration when debugging.

        :return qf: 6x1 numpy array of final joint values. If IK fails to converge
            within the max iterations, the last set of joint angles is still returned.
        :return error_f: 3x1 numpy array of the final positional error.
        :return iters: int, number of iterations taken.
        :return converged: bool, specifies whether the IK solution converged within
            the max iterations.
        """
        ###############################################################################################
        # the following lines of code are data type and error checking. You don't need to understand
        # all of it, but it is helpful to keep.
        if isinstance(q0, np.ndarray):
            q = q0
        elif q0 == None:
            q = np.array([0.0]*self.n)
        elif isinstance(q0, list):
            q = np.array(q0)
        else:
            raise TypeError("Invlid type for initial joint positions 'q0'")

        # Try basic check for if the target is in the workspace.
        # Maximum length of the arm is sum(sqrt(d_i^2 + a_i^2)), distance to target is norm(A_t)
        target_distance = np.linalg.norm(target)
        target_in_reach = target_distance <= self.reach
        if not force:
            assert target_in_reach, "Target outside of reachable workspace!"
        if not target_in_reach:
            print("Target out of workspace, but finding closest solution anyway")

        assert isinstance(K, np.ndarray), "Gain matrix 'K' must be provided as a numpy array"
        ###############################################################################################
        ###############################################################################################

        # you may want to define some functions here to help with operations that you will
        # perform repeatedly in the while loop below. Alternatively, you can also just define
        # them as class functions and use them as self.<function_name>.

        # for example:
        error = self.get_error(q, target)
        iters = 0
        while np.linalg.norm(error) > tol and iters < max_iter:
            iters += 1

            if method == 'pinv':
                # TODO - implement the pseudo-inverse method here
                J = self.jacob(q)
                J_pos = J[0:3, :]  # Extract position part of Jacobian
                J_pseudo_inv = np.linalg.pinv(J_pos)

                delta_q = K @ J_pseudo_inv @ error
                q += delta_q

            elif method == 'J_T':
                # TODO - implement the jacobian transpose method here
                J = self.jacob(q)
                J_pos = J[0:3, :]  # Extract position part of Jacobian
                J_transpose = J_pos.T

                delta_q = J_transpose @ (K @ error)
                q += delta_q

            else:
                raise ValueError("Invalid IK method specified!")

            error = self.get_error(q, target)

            if debug:
                from visualization import VizScene
                import time
                viz = VizScene()
                viz.add_arm(self)
                viz.update(qs=[q])
                viz.add_marker(target)
                if debug_step:
                    input("Press Enter to continue...")
                else:
                    time.sleep(0.1)

        # In this while loop you will update q for each iteration, and update, then
        # your error to see if the problem has converged. You may want to print the error
        # or the "count" at each iteration to help you see the progress as you debug.
        # You may even want to plot an arm initially for each iteration to make sure
        # it's moving in the right direction towards the target.



        # when "while" loop is done, return the relevant info.
        return q, error, iters, iters < max_iter
    
    def get_error(q, target):
            cur_position = arm.fk(q)
            e = cur_position[0:3, 3] - target
            return e





if __name__ == "__main__":
    from visualization import VizScene
    import time

    # Defining a table of DH parameters where each row corresponds to another joint.
    # The order of the DH parameters is [theta, d, a, alpha] - which is the order of operations. 
    # The symbolic joint variables "q" do not have to be explicitly defined here. 
    # This is a two link, planar robot arm with two revolute joints. 
    dh = [[0, 0, 0.3, 0],
          [0, 0, 0.3, 0]]

    # make robot arm (assuming all joints are revolute)
    arm = SerialArm(dh)

    # defining joint configuration
    q = [pi/4.0, pi/4.0]  # 45 degrees and 45 degrees

    # show an example of calculating the entire forward kinematics
    Tn_in_0 = arm.fk(q)
    print("Tn_in_0:\n", Tn_in_0, "\n")

    # show an example of calculating the kinematics between frames 0 and 1
    T1_in_0 = arm.fk(q, index=[0,1])
    print("T1_in 0:\n", T1_in_0, "\n")

    print(arm)

    # now visualizing the coordinate frames that we've calculated
    viz = VizScene()

    viz.add_frame(arm.base, label='base')
    viz.add_frame(Tn_in_0, label="Tn_in_0")
    viz.add_frame(T1_in_0, label="T1_in_0")

    time_to_run = 30
    refresh_rate = 60

    for i in range(refresh_rate * time_to_run):
        viz.update()
        time.sleep(1.0/refresh_rate)
    
    viz.close_viz()
    
# Copy this import into kinematics.py:
from utility import skew

    ## copy this function into the main SerialArm class and complete the TODO below
def Z_shift(self, R: NDArray=np.eye(3), p: NDArray=np.zeros(3,), p_frame: str='i'):
        """
        Z = Z_shift(R, p, p_frame)

        Generates a shifting operator (rotates and translates) to move twists and
        Jacobians from one point to a new point defined by the relative transform
        R and the translation p.

        :param NDArray R: 3x3 array that expresses frame "i" in frame "j" (e.g. R^j_i).
        :param NDArray p: (3,) array (or iterable), the translation from the initial
            Jacobian point to the final point, expressed in the frame as described
            by the next variable.
        :param str p_frame: is either 'i', or 'j'. Allows us to define if "p" is
            expressed in frame "i" or "j", and where the skew symmetrics matrix
            should show up.
        :return Z: 6x6 numpy array, can be used to shift a Jacobian, or a twist.
        """

        # generate our skew matrix
        S = skew(p)

        if p_frame == 'i':
            Z = np.block([[R, np.zeros((3, 3))],
                          [S @ R, R]])
        elif p_frame == 'j':
            Z = np.block([[R, np.zeros((3, 3))],
                          [S @ R, R]])
        else:
            raise ValueError("p_frame must be either 'i' or 'j'")

        return Z
