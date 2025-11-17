import sympy as sp
import numpy as np



##Problem 1
a, b, c, m = sp.symbols('a b c m')
Ixx, Iyy, Izz = sp.symbols('Ixx Iyy Izz')
Ixy, Ixz, Iyz = sp.symbols('Ixy Ixz Iyz')

Ixx = (1/12) * m * (b**2 + c**2)
Iyy = (1/12) * m * (a**2 + c**2)
Izz = (1/12) * m * (a**2 + b**2)
Ixy = 0
Ixz = 0
Iyz = 0

inertia_tensor = sp.Matrix([[Ixx, Ixy, Ixz],
                             [Ixy, Iyy, Iyz],
                                [Ixz, Iyz, Izz]])

print("Inertia Tensor:")
sp.pprint(inertia_tensor)

# Move to the corner of the box

r_com_corner = sp.Matrix([a/2, b/2, c/2])

r_skew = sp.Matrix([[0, -r_com_corner[2], r_com_corner[1]],
                    [r_com_corner[2], 0, -r_com_corner[0]],
                    [-r_com_corner[1], r_com_corner[0], 0]])

inertia_tensor_corner = inertia_tensor + m * r_skew * r_skew.T

print("\nInertia Tensor about corner:")
sp.pprint(inertia_tensor_corner)


