# %% [markdown] 
# # Homework 4 - Key

# %%
import transforms_key_hw04 as tr
import numpy as np
np.set_printoptions(precision=4)


# %% [markdown]
# # Problem 1:

# Check your implementation of rpy, axis, and quaternion. For the given "R_test", you should get the following for RPY, axis/angle, and quaternion represenations:
# $$[\psi, \theta, \phi] = \left[\begin{matrix}1.041\\0.147\\0.5299\end{matrix}\right]$$
# $$[\theta, r] = \left[\begin{matrix}1.13\\0.3574\\0.3574\\0.8629\end{matrix}\right] $$
# $$\mathcal{Q} = \left[\begin{matrix}0.8446\\0.1913\\0.1913\\0.4619\end{matrix}\right] $$

# %%
R_test = tr.rotx(45*np.pi/180.0) @ tr.rotz(45*np.pi/180.0) @ tr.roty(45*np.pi/180.0)

print("Roll, pitch, yaw angles:")
print(tr.R2rpy(R_test))

print("axis/angle representation:")
print(tr.R2axis(R_test))

print("quaternion representation:")
print(tr.R2quat(R_test))


# %% [markdown]
# Now check that rotation about XZY of $\psi=0.787$, $\theta=0.787$, $\phi=0.787$ gives the 
# following:
# $$R =  \left[\begin{matrix}0.4984 & -0.7082 & 0.5\\0.8546 & 0.4984 & -0.1459\\-0.1459 & 0.5 & 0.8537\end{matrix}\right]$$

# the rotation for axis/angle of $\frac{\pi}{2}$ about the $[1, 0, 0]$ axis should give the following rotation matrix:
# $$R =  \left[\begin{matrix}1.0 & 0 & 0\\0 & 0.0 & -1.0\\0 & 1.0 & 0.0\end{matrix}\right]$$

# finally, a quaternion of $\mathcal{Q}=[0.707, 0.707, 0.0, 0.0]$ (assuming an order of $[\nu, \mathbf{\eta}]$) will give the rotation matrix:
# $$R = \left[\begin{matrix}0.9994 & 0 & 0\\0 & -0.000302 & -0.9997\\0 & 0.9997 & -0.000302\end{matrix}\right]$$

# %%
# this should be identical to R_test earlier in our code
R = tr.euler2R(0.787, 0.787, 0.787, 'xzy')
print("R for euler2R was:")
print(R)

# this is just a rotation about the x-axis by pi/2. So checking it should be easy for you. 
R = tr.axis2R(3.14/2, np.array([1, 0, 0]))
print("R for axis2R was:")
print(R)

# quaternions are harder to understand, but after looking at the resulting R matrix,
# can you tell what axis this is quaternion is rotating about? 
R = tr.quat2R(np.array([0.707, 0.707, 0.0, 0.0]))
print("R for quaternion2R was:")
print(R)



# %% [markdown]
# # Problem 2
# Given a $R_y(90^\circ)$ and then $R_z(45^\circ)$, find (a) the axis/angle representation and (b) a quaternion representation.


# %%
R = tr.roty(90*np.pi/180.0) @ tr.rotz(45*np.pi/180.0)

# for part (a)
print("Axis/angle:")
print(tr.R2axis(R))

# for part (b)
print('Quaternion:') 
print(tr.R2quat(R))

# %% [markdown]
# # Problem 3
# - (a) Use "euler2R" function to find RPY($\psi$, $\theta$, $\phi$).
# - (b) A unit vector in the z-direction is just the 3rd column of the rotation matrix.
# - (c) see PDF for algorithm solution

# %%
import sympy as sp
from IPython.display import Math, display
from sympy.physics.vector.printing import vpprint, vlatex


psi, theta, phi = sp.symbols('psi, theta, phi')

# reusing code from HW 02 -> it may make sense to have a symbolic version
# of these functions in our transforms.py file, but I don't yet. 

def rotx(th):
    R = sp.Matrix([[1, 0, 0],
                  [0, sp.cos(th), -sp.sin(th)],
                  [0, sp.sin(th), sp.cos(th)]])
    return R


def roty(th):
    R = sp.Matrix([[sp.cos(th), 0, sp.sin(th)],
                   [0, 1, 0],
                   [-sp.sin(th), 0, sp.cos(th)]])
    return R


def rotz(th):
    R = sp.Matrix([[sp.cos(th), -sp.sin(th), 0],
                   [sp.sin(th), sp.cos(th), 0],
                   [0, 0, 1]])
    return R


R = rotz(psi) @ roty(theta) @ rotx(phi)

# part a)
print('rotation matrix from symbolic variables for roll, pitch, and yaw:')
display(Math(vlatex(sp.simplify(R))))

# part b)
# we could multiply by [0, 0, 1]^T, but we can also just grab the 3rd column
print('unit vector in z-direction:')
#display(R[:, 2])
display(Math(vlatex(R[:,2])))

# %%
