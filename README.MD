### 1. KF Theory

**Prediction**

$\begin{array}{|l|l|l|}
\hline
\text{Univariate} & \text{Univariate} & \text{Multivariate}\\
& \text{(Kalman form)} & \\
\hline
\bar \mu = \mu + \mu_{f_x} & \bar x = x + dx & \bar{\mathbf x} = \mathbf{Fx} + \mathbf{Bu}\\
\bar\sigma^2 = \sigma_x^2 + \sigma_{f_x}^2 & \bar P = P + Q & \bar{\mathbf P} = \mathbf{F_jPF_j}^\mathsf T + \mathbf Q \\
\hline
\end{array}$

**Correction**

$\begin{array}{|l|l|l|}
\hline
\text{Univariate} & \text{Univariate} & \text{Multivariate}\\
& \text{(Kalman form)} & \\
\hline
 & y = z - \bar x & \mathbf y = \mathbf z - \mathbf{H\bar x} \\
 & K = \frac{\bar P}{\bar P+R} &
\mathbf K = \mathbf{\bar{P}H}^\mathsf T (\mathbf{H\bar{P}H}^\mathsf T + \mathbf R)^{-1} \\
\mu=\frac{\bar\sigma^2\, \mu_z + \sigma_z^2 \, \bar\mu} {\bar\sigma^2 + \sigma_z^2} & x = \bar x + Ky & \mathbf x = \bar{\mathbf x} + \mathbf{Ky} \\
\sigma^2 = \frac{\sigma_1^2\sigma_2^2}{\sigma_1^2+\sigma_2^2} & P = (1-K)\bar P &
\mathbf P = (\mathbf I - \mathbf{KH})\mathbf{\bar{P}} \\
\hline
\end{array}$

The details will be different than the univariate filter because these are vectors and matrices, but the concepts are exactly the same: 

-  Use a Gaussian to represent our estimate of the state and error
-  Use a Gaussian to represent the measurement and its error
-  Use a Gaussian to represent the process model
-  Use the process model to predict the next state (the prior)
-  Form an estimate part way between the measurement and the prior

### 2. Demo
#### (1) prediction
input x: [x, y, yaw, v]
state function: 
$$
    x_{t+1} = x_t + v*dt*cos(yaw)\\
    y_{t+1} = y_t + v*dt*sin(yaw)\\
    yaw_{t+1} = yaw_t + \omega*dt\\
    v_{t+1} =  v_{t}\\
$$
so the matrix F is 
$$
F = \begin{bmatrix}
1.0 & 0 &0 &0 \\
0 & 1.0 &0 &0 \\
0 & 0 & 1.0 &0 \\
0 & 0 &0 &0 \\
\end{bmatrix}
$$
and the jacobian  of state function is 
$$
Fj = \begin{bmatrix}
\frac{\partial x_{t+1}}{\partial x} & \frac{\partial x_{t+1}}{\partial y} & \frac{\partial x_{t+1}}{\partial yaw} & \frac{\partial x_{t+1}}{\partial v} \\
\frac{\partial y_{t+1}}{\partial x} & \frac{\partial y_{t+1}}{\partial y} & \frac{\partial y_{t+1}}{\partial yaw} & \frac{\partial y_{t+1}}{\partial v} \\
\frac{\partial yaw}{\partial x} & \frac{\partial yaw}{\partial y} & \frac{\partial yaw}{\partial yaw} & \frac{\partial yaw}{\partial v} \\
\frac{\partial v}{\partial x} & \frac{\partial v}{\partial y} & \frac{\partial v}{\partial yaw} & \frac{\partial v}{\partial v} \\
\end{bmatrix}
$$
hence,  
$$
\frac{dx}{dyaw} = -v*dt*sin(yaw)\\
\frac{dx}{dv} = dt*cos(yaw)\\
\frac{dy}{dyaw} = v*dt*cos(yaw)\\
\frac{dy}{dv} = dt*sin(yaw)\\
$$
#### (2)update
if we use GPS as a positioning observation
the measure function is 
$$
z = \begin{bmatrix}
1 & 0 & 0 & 0 \\
0 & 1 & 0 & 0 \\
\end{bmatrix} 
\begin{bmatrix}
x \\ y \\ yaw \\ v \\
\end{bmatrix} 
$$
after calc Jacobian uesd equal object with Fj,we can finish KF as folow :
- S = Hj * P_Pred * Hj.T + R 
- K = P_pred * Hj.T  *np.linalg.inv(S)
- x_est = x_pred + K * y 
- PEst = (np.eye(len(x_est)) - K*$H_j$) $*$ P_pred 

#### Ref:
- [PROBABILISTIC ROBOTICS](http://www.probabilistic-robotics.org/)

#### Requirements

- Python 3.6.x

- numpy

- matplotlib

