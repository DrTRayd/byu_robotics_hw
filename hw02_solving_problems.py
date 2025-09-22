import numpy as np
import transforms as tr
import sympy as sp
from visualization import VizScene



#problem 6 workout

# R21 = np.array([[1, 0, 0],
#                 [1, 0.5, -np.sqrt(3)/2],
#                 [0, np.sqrt(3)/2, 0.5]])

# R31 = np.array([[0, 0, -1],
#                 [0, 1, 0],
#                 [1, 0, 0]])

# R12 = R21.T

# print(R12)

# R32 = R12 @ R31

# print(R32)

#________________________________________________
#problem 7 workout

# theta, phi, psi, x = sp.symbols('theta phi psi x')

# rotx = sp.Matrix([[1, 0, 0],
#                    [0, sp.cos(x), -sp.sin(x)],
#                    [0, sp.sin(x), sp.cos(x)]])

# roty = sp.Matrix([[sp.cos(x), 0, sp.sin(x)],
#                      [0, 1, 0],
#                         [-sp.sin(x), 0, sp.cos(x)]])

# rotz = sp.Matrix([[sp.cos(x), -sp.sin(x), 0],
#                    [sp.sin(x), sp.cos(x), 0],
#                    [0, 0, 1]])

# R = rotx.subs(x, theta) @ roty.subs(x, phi) @ rotz.subs(x, sp.pi) @ roty.subs(x, -phi) @ rotx.subs(x, -theta)

# sp.pprint(sp.simplify(R))

#________________________________________________
#problem 8 workout

# R = tr.rotz(np.pi/2) @ tr.roty(0) @ tr.rotz(np.pi/4)

# R_sympy = sp.Matrix(R)
# R_fraction = R_sympy.applyfunc(sp.nsimplify)  # Converts elements to fractions
# sp.pprint(R_fraction)

# # Create a 4x4 identity matrix
# R_4x4 = np.eye(4)

# # Replace the top-left 3x3 block with the original matrix
# R_4x4[:3, :3] = R

# viz = VizScene()
# viz.add_frame(np.eye(4), label='world', axes_label='w')
# viz.add_frame(R_4x4, label='frame1', axes_label='1')

# viz.hold()



#__________
#Problem 9 workout

# this cell will fail if you haven't correctly installed the libraries in the "requiremnts.txt" file
import numpy as np
import time
from visualization import VizScene

Tw_to_frame1 = np.eye(4)

viz = VizScene()
viz.add_frame(np.eye(4), label='world', axes_label='w')
viz.add_frame(Tw_to_frame1, label='frame1', axes_label='1')

time_to_run = 100
refresh_rate = 60
t = 0
start = time.time()
while t < time_to_run:
    t = time.time() - start

    # you can play with omega and p to see how they affect the frame
    omega = np.pi/2
    R = np.array([[np.cos(omega*t), -np.sin(omega*t), 0],
                  [np.sin(omega*t), np.cos(omega*t), 0],
                  [0, 0, 1]])
    p = np.array([1, 0, 0])

    Tw_to_frame1[:3,:3] = R
    Tw_to_frame1[:3,-1] = p
    viz.update(As=[np.eye(4), Tw_to_frame1])

viz.close_viz() # could use viz.hold() to keep it open until manually closed