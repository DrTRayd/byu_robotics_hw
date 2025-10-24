import numpy as np

# z0_0 = np.array([0,0,1])

# t0_0 = np.array([0,0,1.5])

# cross = np.cross(z0_0, t0_0)
# print("z0_0:", z0_0)
# print("t0_0:", t0_0)
# print("Cross:", cross)

J = np.array ([[0, 4.5, 3, 0, -1.5, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 1, 1, 0, -1, 0], [1, 0, 0, 1, 0, 1]])
J_T = J.T
forces = np.array([100, 0, 0, 0, 0, 0])
print("J:")
print(J)
print(" ")
print("J_T:")
print(J_T)

torques = J_T @ forces

print(" ")
print("Torques:")
print(torques)