# -- coding: utf-8 --

import numpy as np
import math
import matplotlib.pyplot as plt

# estimation parameter of ekf
Q = np.diag([0.1,0.1,math.radians(1.0),1.0])**2
R = np.diag([1.0,math.radians(40.0)])**2

# simulation parameter
Qsim = np.diag([0.5, 0.5])**2
Rsim = np.diag([1.0, math.radians(30.0)])**2

DT = 0.1 # TIME tick[s]
SIM_TIME = 50.0
show_animation=True

def calc_input():
    v = 1.0 #[m/s]
    yawrate = 0.1 #[rad/s]
    u = np.matrix([v,yawrate]).T
    return u

def motion_model(x,u):
    F = np.matrix([[1.0,0,0,0],
                 [0,1.0,0,0],
                 [0,0,1.0,0],
                 [0,0,0,0]])
    B = np.matrix([[DT*math.cos(x[2,0]),0],
                   [DT*math.sin(x[2,0]),0],
                   [0.0,DT],
                   [1.0,0.0]])
    x = F*x + B*u
    
    return x

def observation(xTrue,xd,u):
    xTrue = motion_model(xTrue,u)
    # add noise to gps x-y
    zx = xTrue[0,0] + np.random.randn()*Qsim[0,0]
    zy = xTrue[1,0] + np.random.randn()*Qsim[1,1]
    z = np.matrix([zx,zy])
    # add noise to input
    ud1 = u[0,0] + np.random.randn()*Rsim[0,0]
    ud2 = u[1,0] + np.random.randn()*Rsim[1,1]
    ud = np.matrix([ud1,ud2]).T
    
    xd = motion_model(xd,ud)
    return xTrue, z, xd, ud

def observation_model(x):
    # observation model
    H = np.matrix([[1,0,0,0],
                  [0,1,0,0]])
    z = H * x
    return z

def jacobF(x,u):
    """
    jacobian of motion model
    mption model
    x_{t+1} = x_t + v*dt*cos(yaw)
    y_{t+1} = y_t + v*dt*sin(yaw)
    yaw_{t+1} = yaw_t + omega*dt
    v_{t+1} =  v_{t}
    
    so 
    dx/dyaw = -v*dt*sin(yaw)
    dx/dv = dt*cos(yaw)
    dy/dyaw = v*dt*cos(yaw)
    dy/dv = dt*sin(yaw)
    """
    yaw = x[2,0]
    v = u[0,0]
    jF = np.matrix([[1.0,0.0,-DT*v*math.sin(yaw),DT*math.cos(yaw)],
                    [0.0,1.0,DT*v*math.cos(yaw),DT*math.sin(yaw)],
                    [0.0,0.0,1.0,0.0],
                    [0.0,0.0,0.0,1.0]])
    return jF

def jacobH(x):
    # jacobian of observation model
    jH = np.matrix([
        [1,0,0,0],
        [0,1,0,0]
    ])
    return jH

def ekf_estimation(xEst,PEst,z,u):
    #predict
    xPred = motion_model(xEst,u)
    jF = jacobF(xPred,u)
    PPred = jF*PEst*jF.T+Q 
    #update
    jH = jacobH(xPred)
    zPred = observation_model(xPred)
    y = z.T - zPred
    S = jH * PPred * jH.T + R
    K = PPred * jH.T  *np.linalg.inv(S)
    xEst = xPred + K * y
    PEst = (np.eye(len(xEst)) - K*jH) * PPred
    return xEst,PEst
    
def plot_cov(xEst,PEst):
    Pxy = PEst[0:2,0:2]
    eigval,eigvec = np.linalg.eig(Pxy)
    if eigval[0] >= eigval[1]:
        bigind = 0
        smallind =1
    else:
        bigind = 1
        smallind = 0 
        
    t = np.arange(0,2*math.pi + 0.1, 0.1)
    a = math.sqrt(eigval[bigind])
    b = math.sqrt(eigval[smallind])
    x = [a*math.cos(it) for it in t]
    y = [b*math.sin(it) for it in t]
    angle = math.atan2(eigvec[bigind,1],eigvec[bigind,0])
    R = np.matrix([[math.cos(angle),math.sin(angle)],
                   [-math.sin(angle),math.cos(angle)]])
    fx = R* np.matrix([x,y])
    px = np.array(fx[0,:]+xEst[0,0].flatten())
    py = np.array(fx[1,:]+xEst[1,0].flatten())
    plt.plot(px,py,'--r')

def main():
    time =0.0
    # state vector [x, y, yaw, v]
    xEst = np.matrix(np.zeros((4,1)))
    xTrue = np.matrix(np.zeros((4,1)))
    PEst = np.eye(4)
    xDR = np.matrix(np.zeros((4,1)))#dead reckoning
    #history
    hxEst = xEst
    hxTrue = xTrue
    hxDR = xTrue
    hz = np.zeros((1,2))
    while SIM_TIME >= time:
        time += DT
        u = calc_input()
        xTrue,z,xDR,ud = observation(xEst,xDR,u)
        xEst,PEst = ekf_estimation(xEst,PEst,z,ud)
        #store data history
        hxEst = np.hstack((hxEst,xEst))
        hxDR = np.hstack((hxDR,xDR))
        hxTrue = np.hstack((hxTrue,xTrue))
        hz = np.vstack((hz,z))
        if show_animation:
            plt.cla()
            plt.plot(hz[:,0],hz[:,1],'.g')
            plt.plot(np.array(hxTrue[0,:]).flatten(),
                    np.array(hxTrue[1,:]).flatten(),'-b')
            plt.plot(np.array(hxDR[0,:]).flatten(),
                    np.array(hxDR[1,:]).flatten(),'-k')
            plt.plot(np.array(hxEst[0,:]).flatten(),
                    np.array(hxEst[1,:]).flatten(),'-r')
            plot_cov(xEst,PEst)
            plt.axis('equal')
            plt.grid(True)
            plt.pause(0.001)

if __name__ == "__main__":
    main()            