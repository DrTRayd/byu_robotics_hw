import kinematics as kin
# import transforms_key_hw04 as tr
from visualization import VizScene
import numpy as np
from numpy import pi
import matplotlib.pyplot as plt




from scipy.io import savemat, loadmat


dh = [[   0, 270.35, 69, -pi/2],
     [pi/2,      0,  0,  pi/2],
     [   0, 364.35, 69, -pi/2],
     [   0,      0,  0,  pi/2],
     [   0, 374.29, 10, -pi/2],
     [   0,      0,  0,  pi/2],
     [   0,229.525,  0,     0],]
# dh = [[   0, 270.35, 69, -pi/2],
#       [pi/2,      0,  0,  pi/2],
#       [   0, 364.35, 69, -pi/2],
#       [  pi,      0,  0, -pi/2],
#       [   0, 374.29,-10,  pi/2],
#       [   0,      0,  0, -pi/2],
#       [   0,229.525,  0,     0],]




# base = [[-0.7071,-0.7071,    0, 0.06353],
#         [ 0.7071,-0.7071,    0, -0.2597],
#         [      0,      0,    1,   0.119],
#         [      0,      0,    0,       1],]
base = [[      1,      0,    0, 0.06353],
       [      0,      1,    0, -0.2597],
       [      0,      0,    1,   0.119],
       [      0,      0,    0,       1],]
base = np.array(base)


tip = [[0,0,-1,0],
      [0,1,0,0],
      [1,0,0,0],
      [0,0,0,1],]
arm = kin.SerialArm(dh,['r','r','r','r','r','r','r'],base=base,tip=tip)


# mat_input = loadmat('part3_trial00.mat')
# t = mat_input['t']
# wrench = np.zeros((6,t.shape[1],10))


# for i in range(10):
#     mat_input = loadmat(f'part3_trial0{i}.mat')
#     q = mat_input['q']
#     qdot = mat_input['q_dot']
#     t = mat_input['t']


#     for j in range(len(q)):
#         wrench[:,j,i] = arm.jacob(q[j,:]) @ qdot[j,:]


# wrench_avg = np.mean(wrench,2)
# wrench_std = np.std(wrench,2)


# ci = 2*abs(np.abs(wrench_std))


# fig, ax = plt.subplots(2,1)
# ax[0].plot(t.T,wrench_avg[0,:].T/1000)
# ax[0].fill_between(t.T, (wrench_avg[0,:]-ci[0,:]).T, (wrench_avg[0,:]+ci[0,:]).T, color='blue', alpha=0.2, label='95% CI')


# ax[1].plot(t.T,wrench_avg[3:,:].T)


# ax[0].set_ylabel('Linear Velocity (m/s)')
# ax[1].set_xlabel('Time (s)')
# ax[1].set_ylabel('Angular Velocity (rad/s)')
# plt.show()












# dh = [[0,270.35,69,pi/2],
#       [pi/2,0,0,pi/2],
#       [0,364.35,-69,-pi/2],
#       [pi,0,0,-pi/2],
#       [0,374.29,-10,pi/2],
#       [0,0,0,-pi/2],
#       [0,229.525,0,0],]




# q = [0,0,0,0,0,0,0]




q =  [ 0.83218458,-0.40420394,-2.88733534, 0.51043211,-3.04495186, 1.60684488, 0.55568454]


something = arm.fk(q)
print(something[0:3,3]/1000)


q =  [-8.55194289e-02,-2.37000032e-01,-2.96096642e+00,-1.53398079e-03,-3.04533536e+00,1.81124782e+00,7.69291365e-01]


something = arm.fk(q)
print(something[0:3,3]/1000)


q = [ 0.54993211,0.07017962,-2.98359263,0.60975736,-3.04571885,1.66321867,2.26530613]


something = arm.fk(q)
print(something[0:3,3]/1000)


q =  [0.96257294,-0.43756802,-2.98320914,1.00283994,-3.02692759,1.08912636,3.01465574]


something = arm.fk(q)
print(something[0:3,3]/1000)




# viz = VizScene()
# viz.grid
# viz.add_arm(arm,True,)
# viz.update(q)
# viz.hold()




