## The template python file for hw09 can be a bit confusing.
## Please feel free to only use whatever makes sense to you and
## delete the rest if you don't find it helpful.

# %% [markdown]
# # Homework 9

# %%
import dynamics as dyn
from visualization import VizScene
from scipy.integrate import solve_ivp
from scipy.io import loadmat
from matplotlib import pyplot as pl

import numpy as np


# %% [markdown]
# # Problem #1

# %%
# set up model

# defining kinematic parameters for three-link planar robot
dh = [[0, 0, 0.2, 0],
      [0, 0, 0.2, 0],
      [0, 0, 0.2, 0]]

joint_type = ['r', 'r', 'r']

link_masses = [1, 0.5, 0.3]

# defining three different centers of mass, one for each link
r_coms = [np.array([-0.1, 0, 0]),
          np.array([-0.1, 0, 0]),
          np.array([-0.1, 0, 0])]

# all terms except Izz are zero because they don't matter in the
# equations, Ixx, and Iyy are technically non-zero, we just don't
# rotate about those axes so it doesn't matter.
link_inertias = [np.diag([0, 0, 0.1]),
                 np.diag([0, 0, 0.1]),
                 np.diag([0, 0, 0.1])]

# the viscous friction coefficients for B*q_dot are:
B = np.diag([0.8, 0.3, 0.2])  # but you will need to either include
                              # these in your dynamic arm constructor,
                              # or just use them as a global variable
                              # in your EOM function for numerical
                              # integration.


# my assumption is that you are using your RNE dynamics
arm = dyn.SerialArmDyn(dh,
                       jt=joint_type,
                       mass=link_masses,
                       r_com=r_coms,
                       link_inertia=link_inertias)


# %% [markdown]
# part a) - calculate joint torques here using your RNE, or E-L equations
# %%
data = loadmat('desired_accel.mat')

t = data['t'].squeeze() # remove singleton dimensions
q = data['q'] # probably want to figure out shape of this, relates to axis of np.gradient

# TODO you will need to take the derivative of q to get qd and qdd,
# you can use np.gradient with q and the time vector "t" but you will need
# to specify the axis (0 or 1) that you are taking the gradient along,
# e.g. np.gradient(q, t, axis=)
qd = np.gradient(q, t, axis=0)
qdd = np.gradient(qd, t, axis=0)


# TODO - calc torques for every time step given q, qd, qdd
torque = np.zeros_like(q)  # pre-allocate torque array
for i in range(len(t)):
    # TODO - calculate torque at time index i using RNE
    tau, _ = arm.rne(q[i,:], qd[i,:], qdd[i,:])
    torque[i,:] = tau

# If you have a vector of all torques called "torque",
# you can use the following code to plot:
pl.figure()
for i in range(arm.n):
    pl.subplot(arm.n, 1, i+1)
    pl.plot(t, torque[:,i])
    pl.ylabel('joint '+str(i+1))
pl.xlabel('time (s)')
pl.tight_layout()
pl.show()



# %% [markdown]
# part b) - perform numerical integration as specified in  problem statement


# %%

# these are simplified equations of motion (since we assume that motor torque = 0)
def robot_grav(t, x):
    x_dot = np.zeros(2*arm.n)

    # TODO - define your EOM function here (that returns x_dot)
    x = x.reshape(2*arm.n, )
    q = x[arm.n:]
    qd = x[0:arm.n]

    # calculate gravity and coriolis terms
    G = arm.get_G(q, g = np.array([0, 0, -9.81]))
    C = arm.get_C(q, qd)
    # calculate qdd
    qdd = -np.linalg.inv(arm.get_M(q)) @ (C + G + B @ qd)
    x_dot[0:arm.n] = qdd
    x_dot[arm.n:] = qd

    return x_dot

# TODO - perform numerical integration here using "solve_ivp".
# When finished, you can use the plotting code below to help you.

# initial conditions - starting at rest, all joints at zero position
x0 = np.zeros(2*arm.n)
# time span for simulation
t_span = (0, 5)  # simulate for 5 seconds
sol = solve_ivp(robot_grav, t_span, x0, t_eval = np.linspace(t_span[0], t_span[1], 500))


## NOTE: In all of the plotting code below, I assume x = [qd, q]. If you used
## x = [q, qd], you will need to change the indexing in the plotting code below.

# making an empty figure
fig = pl.figure()

# plotting the time vector "t" versus the solution vector for
# the three joint positions, entry 3-6 in sol.y
pl.plot(sol.t, sol.y[arm.n:].T)
pl.ylabel('joint positions (rad)')
pl.xlabel('time (s)')
pl.title('three-link robot falling in gravity field')
pl.show()


# now show the actual robot being simulated in pyqtgraph, this will only
# work if you have found the integration solution
# %%
# visualizing the robot acting under gravity
viz = VizScene()
viz.add_arm(arm)

for i in range(len(sol.t)-1):
    viz.update(qs=[sol.y[arm.n:,i]])
    viz.hold(t[i+1]-t[i])
viz.close_viz()


# %% [markdown]
# # Problem #2

# Define the dynamics function (x_dot = f(x,u)) for integration
def eom(t, x, controller):
    x = x.reshape(2*arm.n)
    q  = x[arm.n:]
    qd = x[:arm.n]

    tau = controller(t, q, qd)       # <-- controller inserted here

    M = arm.get_M(q)
    C = arm.get_C(q, qd)
    G = arm.get_G(q, g=np.array([0, 0, -9.81]))

    qdd = np.linalg.inv(M) @ (tau - C - G - B @ qd)

    x_dot = np.zeros_like(x)
    x_dot[:arm.n] = qdd
    x_dot[arm.n:] = qd
    return x_dot


# you can define any q_des, qd_des, and qdd_des you want, but feel free to use this
# code below if it makes sense to you. I'm just defining q as a function of time
# and then taking the symbolic derivative.
import sympy as sp

t = sp.symbols('t')
q_des_sp = sp.Matrix([sp.cos(2*sp.pi*t),
                      sp.cos(2*sp.pi*2*t),
                      sp.cos(2*sp.pi*3*t)])
qd_des_sp = q_des_sp.diff(t)
qdd_des_sp = qd_des_sp.diff(t)

# turn them into numpy functions so that they are faster and return
# the right data type. Now we can call these functions at any time "t"
# in the "eom" function.
q_des = sp.lambdify(t, q_des_sp, modules='numpy')
qd_des = sp.lambdify(t, qd_des_sp, modules='numpy')
qdd_des = sp.lambdify(t, qdd_des_sp, modules='numpy')

# %%
# TODO define three different control functions and numerically integrate for each one.
# If "sol" is output from simulation, plotting can look something like this:

# a) Implement a PD controller for each joint, add gravity compensation if necessary
def pd_control(t, q, qd):
    # PD gains
    Kp = np.diag([100, 100, 100])
    Kd = np.diag([20, 20, 20])
    # Flatten
    qd_t = qd_des(t).flatten()
    q_t = q_des(t).flatten()
    # compute control torque
    tau = Kp @ (q_t - q) + Kd @ (qd_t - qd) + arm.get_G(q, g = np.array([0, 0, -9.81]))
    return tau


# b) Implement a feedforward controller with PD control
def feedforward_control(t, q, qd):
    # PD gains
    Kp = np.diag([100, 100, 100])
    Kd = np.diag([20, 20, 20])
    # Flatten
    qd_t = qd_des(t).flatten()
    q_t = q_des(t).flatten()
    qdd_t = qdd_des(t).flatten()
    # compute control torque
    tau = (arm.get_M(q) @ qdd_t +
           arm.get_C(q, qd) +
           arm.get_G(q, g = np.array([0, 0, -9.81])) +
           Kp @ (q_t - q) +
           Kd @ (qd_t - qd)
           - B @ qd)
    return tau

# c) Implement a computed torque controller with PD control
def computed_torque_control(t, q, qd):
    # PD gains
    Kp = np.diag([100, 100, 100])
    Kd = np.diag([20, 20, 20])
    #Flatten 
    qd_t = qd_des(t).flatten()
    q_t = q_des(t).flatten()
    qdd_t = qdd_des(t).flatten()
    # compute control torque
    tau = (arm.get_M(q) @ (qdd_t + Kd @ (qd_t - qd) + Kp @ (q_t - q)) +
           arm.get_C(q, qd) +
           arm.get_G(q, g = np.array([0, 0, -9.81])) - B @ qd)
    return tau

num_sec = 5
time = np.linspace(0, num_sec, num=100*num_sec)
x0 = np.zeros(2*arm.n)
sol = solve_ivp(lambda t, x: eom(t, x, computed_torque_control), (0, num_sec), x0, t_eval=time)

pl.figure()
title = "Computed Torque Control"
for i in range(arm.n):
    pl.subplot(arm.n, 1, i+1)
    pl.plot(sol.t, sol.y[arm.n+i,:].T, label='actual')
    pl.plot(sol.t, q_des(time)[i,:].T, '--', label='commanded')
    pl.legend()
    pl.ylabel('joint '+str(i+1))
pl.xlabel('time (s)')
pl.suptitle(title)
pl.tight_layout()
pl.subplots_adjust(top=0.88)
pl.show()

# %% [markdown]
# # Problem 3 — Tracking Performance with Model Errors

percent_err = 0.10

# masses of each link with some error.
link_masses_error = [np.random.uniform(low = link_masses[0]*(1-percent_err), high = link_masses[0]*(1+percent_err)),
               np.random.uniform(low = link_masses[1]*(1-percent_err), high = link_masses[1]*(1+percent_err)),
               np.random.uniform(low = link_masses[2]*(1-percent_err), high = link_masses[2]*(1+percent_err))]

# defining three different centers of mass, one for each link
r_coms_error = [np.array([np.random.uniform(low = -0.1*(1+percent_err), high = -0.1*(1-percent_err)), 0, 0]),
          np.array([np.random.uniform(low = -0.1*(1+percent_err), high = -0.1*(1-percent_err)), 0, 0]),
          np.array([np.random.uniform(low = -0.1*(1+percent_err), high = -0.1*(1-percent_err)), 0, 0])]

#Link inertias remain the same

# Noisy arm used ONLY by controller
noisy_arm = dyn.SerialArmDyn(
    dh,
    jt=joint_type,
    mass=link_masses_error,
    r_com=r_coms_error,
    link_inertia=link_inertias
)

def sim_eom(t, x, controller):
    x = x.reshape(2*arm.n)
    q  = x[arm.n:]
    qd = x[:arm.n]

    # controller uses noisy model (passed in)
    tau = controller(t, q, qd)

    # REAL dynamics
    M = arm.get_M(q)
    C = arm.get_C(q, qd)
    G = arm.get_G(q, g=np.array([0, 0, -9.81]))

    qdd = np.linalg.inv(M) @ (tau - C - G - B @ qd)

    xdot = np.zeros_like(x)
    xdot[:arm.n] = qdd
    xdot[arm.n:] = qd
    return xdot

def noisy_eom(t, x, controller):
    x = x.reshape(2*arm.n)
    q  = x[arm.n:]
    qd = x[:arm.n]

    tau = controller(t, q, qd)       # <-- controller inserted here

    M = noisy_arm.get_M(q)
    C = noisy_arm.get_C(q, qd)
    G = noisy_arm.get_G(q, g=np.array([0, 0, -9.81]))

    qdd = np.linalg.inv(M) @ (tau - C - G - B @ qd)

    x_dot = np.zeros_like(x)
    x_dot[:arm.n] = qdd
    x_dot[arm.n:] = qd
    return x_dot

import sympy as sp
t_sym = sp.symbols('t')

q_des_sp = sp.Matrix([
    sp.cos(2*sp.pi*t_sym),
    sp.cos(2*sp.pi*2*t_sym),
    sp.cos(2*sp.pi*3*t_sym)
])
qd_des_sp  = q_des_sp.diff(t_sym)
qdd_des_sp = qd_des_sp.diff(t_sym)

q_des  = sp.lambdify(t_sym, q_des_sp,  modules='numpy')
qd_des = sp.lambdify(t_sym, qd_des_sp, modules='numpy')
qdd_des = sp.lambdify(t_sym, qdd_des_sp, modules='numpy')

def pd_control_noisy(t, q, qd):
    Kp = np.diag([100, 100, 100])
    Kd = np.diag([20, 20, 20])

    q_t  = q_des(t).flatten()
    qd_t = qd_des(t).flatten()

    return (Kp @ (q_t - q) +
            Kd @ (qd_t - qd) +
            noisy_arm.get_G(q, g=np.array([0,0,-9.81])))

def feedforward_control_noisy(t, q, qd):
    Kp = np.diag([100,100,100])
    Kd = np.diag([20,20,20])

    q_t   = q_des(t).flatten()
    qd_t  = qd_des(t).flatten()
    qdd_t = qdd_des(t).flatten()

    return (noisy_arm.get_M(q) @ qdd_t +
            noisy_arm.get_C(q, qd) +
            noisy_arm.get_G(q, g=np.array([0,0,-9.81])) +
            Kp @ (q_t - q) +
            Kd @ (qd_t - qd))

def computed_torque_control_noisy(t, q, qd):
    Kp = np.diag([100,100,100])
    Kd = np.diag([20,20,20])

    q_t   = q_des(t).flatten()
    qd_t  = qd_des(t).flatten()
    qdd_t = qdd_des(t).flatten()

    v = qdd_t + Kd @ (qd_t - qd) + Kp @ (q_t - q)
    return (noisy_arm.get_M(q) @ v +
            noisy_arm.get_C(q, qd) +
            noisy_arm.get_G(q, g=np.array([0,0,-9.81])))


num_sec = 5
time = np.linspace(0, num_sec, num=100*num_sec)
x0 = np.zeros(2*arm.n)

# nominal: controller uses TRUE model (Problem 2 controller)
sol_nominal = solve_ivp(
    lambda t, x: sim_eom(t, x, computed_torque_control)*1.25+.25,
    (0, num_sec), x0, t_eval=time
)

# noisy: controller uses incorrect (noisy) model
sol_error = solve_ivp(
    lambda t, x: noisy_eom(t, x, computed_torque_control_noisy)*1.25+.25,
    (0, num_sec), x0, t_eval=time
)

# reshape commanded positions safely
qd_cmd = np.array(q_des(time)) 
qd_cmd = qd_cmd.reshape(arm.n, -1)  # ensure shape is (3, 500)

pl.figure()
title = "PD + G control with computed torque control"

for i in range(arm.n):
    pl.subplot(arm.n, 1, i+1)

    pl.plot(sol_nominal.t, sol_nominal.y[arm.n+i,:], label='actual no error')
    pl.plot(sol_error.t,  sol_error.y[arm.n+i,:],  label='actual with error')
    pl.plot(time, qd_cmd[i], '--', label='commanded')  # now shapes match

    pl.ylabel(f'joint {i+1}')
    pl.legend()

pl.xlabel("time (s)")
pl.suptitle(title)
pl.tight_layout()
pl.subplots_adjust(top=0.88)
pl.show()

# %%
