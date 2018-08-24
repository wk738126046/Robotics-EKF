
### 1. KF Theory

**Prediction**

<img src="https://latex.codecogs.com/png.latex?\begin{array}{|l|l|l|}&space;\hline&space;\text{Univariate}&space;&&space;\text{Univariate}&space;&&space;\text{Multivariate}\\&space;&&space;\text{(Kalman&space;form)}&space;&&space;\\&space;\hline&space;\bar&space;\mu&space;=&space;\mu&space;&plus;&space;\mu_{f_x}&space;&&space;\bar&space;x&space;=&space;x&space;&plus;&space;dx&space;&&space;\bar{\mathbf&space;x}&space;=&space;\mathbf{Fx}&space;&plus;&space;\mathbf{Bu}\\&space;\bar\sigma^2&space;=&space;\sigma_x^2&space;&plus;&space;\sigma_{f_x}^2&space;&&space;\bar&space;P&space;=&space;P&space;&plus;&space;Q&space;&&space;\bar{\mathbf&space;P}&space;=&space;\mathbf{F_jPF_j}^\mathsf&space;T&space;&plus;&space;\mathbf&space;Q&space;\\&space;\hline&space;\end{array}" title="\begin{array}{|l|l|l|} \hline \text{Univariate} & \text{Univariate} & \text{Multivariate}\\ & \text{(Kalman form)} & \\ \hline \bar \mu = \mu + \mu_{f_x} & \bar x = x + dx & \bar{\mathbf x} = \mathbf{Fx} + \mathbf{Bu}\\ \bar\sigma^2 = \sigma_x^2 + \sigma_{f_x}^2 & \bar P = P + Q & \bar{\mathbf P} = \mathbf{F_jPF_j}^\mathsf T + \mathbf Q \\ \hline \end{array}" />

**Correction**

<img src="https://latex.codecogs.com/png.latex?\begin{array}{|l|l|l|}&space;\hline&space;\text{Univariate}&space;&&space;\text{Univariate}&space;&&space;\text{Multivariate}\\&space;&&space;\text{(Kalman&space;form)}&space;&&space;\\&space;\hline&space;&&space;y&space;=&space;z&space;-&space;\bar&space;x&space;&&space;\mathbf&space;y&space;=&space;\mathbf&space;z&space;-&space;\mathbf{H\bar&space;x}&space;\\&space;&&space;K&space;=&space;\frac{\bar&space;P}{\bar&space;P&plus;R}&&space;\mathbf&space;K&space;=&space;\mathbf{\bar{P}H}^\mathsf&space;T&space;(\mathbf{H\bar{P}H}^\mathsf&space;T&space;&plus;&space;\mathbf&space;R)^{-1}&space;\\&space;\mu=\frac{\bar\sigma^2\,&space;\mu_z&space;&plus;&space;\sigma_z^2&space;\,&space;\bar\mu}&space;{\bar\sigma^2&space;&plus;&space;\sigma_z^2}&space;&&space;x&space;=&space;\bar&space;x&space;&plus;&space;Ky&space;&&space;\mathbf&space;x&space;=&space;\bar{\mathbf&space;x}&space;&plus;&space;\mathbf{Ky}&space;\\&space;\sigma^2&space;=&space;\frac{\sigma_1^2\sigma_2^2}{\sigma_1^2&plus;\sigma_2^2}&space;&&space;P&space;=&space;(1-K)\bar&space;P&space;&&space;\mathbf&space;P&space;=&space;(\mathbf&space;I&space;-&space;\mathbf{KH})\mathbf{\bar{P}}&space;\\&space;\hline&space;\end{array}" title="\begin{array}{|l|l|l|} \hline \text{Univariate} & \text{Univariate} & \text{Multivariate}\\ & \text{(Kalman form)} & \\ \hline & y = z - \bar x & \mathbf y = \mathbf z - \mathbf{H\bar x} \\ & K = \frac{\bar P}{\bar P+R}& \mathbf K = \mathbf{\bar{P}H}^\mathsf T (\mathbf{H\bar{P}H}^\mathsf T + \mathbf R)^{-1} \\ \mu=\frac{\bar\sigma^2\, \mu_z + \sigma_z^2 \, \bar\mu} {\bar\sigma^2 + \sigma_z^2} & x = \bar x + Ky & \mathbf x = \bar{\mathbf x} + \mathbf{Ky} \\ \sigma^2 = \frac{\sigma_1^2\sigma_2^2}{\sigma_1^2+\sigma_2^2} & P = (1-K)\bar P & \mathbf P = (\mathbf I - \mathbf{KH})\mathbf{\bar{P}} \\ \hline \end{array}" />

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

<img src="https://latex.codecogs.com/png.latex?x_{t&plus;1}&space;=&space;x_t&space;&plus;&space;v*dt*cos(yaw)\\&space;y_{t&plus;1}&space;=&space;y_t&space;&plus;&space;v*dt*sin(yaw)\\&space;yaw_{t&plus;1}&space;=&space;yaw_t&space;&plus;&space;\omega*dt\\&space;v_{t&plus;1}&space;=&space;v_{t}\\" title="x_{t+1} = x_t + v*dt*cos(yaw)\\ y_{t+1} = y_t + v*dt*sin(yaw)\\ yaw_{t+1} = yaw_t + \omega*dt\\ v_{t+1} = v_{t}\\" />

so the matrix F is 

<a href="https://www.codecogs.com/eqnedit.php?latex=F&space;=&space;\begin{bmatrix}&space;1.0&space;&&space;0&space;&0&space;&0&space;\\&space;0&space;&&space;1.0&space;&0&space;&0&space;\\&space;0&space;&&space;0&space;&&space;1.0&space;&0&space;\\&space;0&space;&&space;0&space;&0&space;&0&space;\\&space;\end{bmatrix}" target="_blank"><img src="https://latex.codecogs.com/png.latex?F&space;=&space;\begin{bmatrix}&space;1.0&space;&&space;0&space;&0&space;&0&space;\\&space;0&space;&&space;1.0&space;&0&space;&0&space;\\&space;0&space;&&space;0&space;&&space;1.0&space;&0&space;\\&space;0&space;&&space;0&space;&0&space;&0&space;\\&space;\end{bmatrix}" title="F = \begin{bmatrix} 1.0 & 0 &0 &0 \\ 0 & 1.0 &0 &0 \\ 0 & 0 & 1.0 &0 \\ 0 & 0 &0 &0 \\ \end{bmatrix}" /></a>

and the jacobian  of state function is 

<a href="https://www.codecogs.com/eqnedit.php?latex=Fj&space;=&space;\begin{bmatrix}&space;\frac{\partial&space;x_{t&plus;1}}{\partial&space;x}&space;&&space;\frac{\partial&space;x_{t&plus;1}}{\partial&space;y}&space;&&space;\frac{\partial&space;x_{t&plus;1}}{\partial&space;yaw}&space;&&space;\frac{\partial&space;x_{t&plus;1}}{\partial&space;v}&space;\\&space;\frac{\partial&space;y_{t&plus;1}}{\partial&space;x}&space;&&space;\frac{\partial&space;y_{t&plus;1}}{\partial&space;y}&space;&&space;\frac{\partial&space;y_{t&plus;1}}{\partial&space;yaw}&space;&&space;\frac{\partial&space;y_{t&plus;1}}{\partial&space;v}&space;\\&space;\frac{\partial&space;yaw}{\partial&space;x}&space;&&space;\frac{\partial&space;yaw}{\partial&space;y}&space;&&space;\frac{\partial&space;yaw}{\partial&space;yaw}&space;&&space;\frac{\partial&space;yaw}{\partial&space;v}&space;\\&space;\frac{\partial&space;v}{\partial&space;x}&space;&&space;\frac{\partial&space;v}{\partial&space;y}&space;&&space;\frac{\partial&space;v}{\partial&space;yaw}&space;&&space;\frac{\partial&space;v}{\partial&space;v}&space;\\&space;\end{bmatrix}" target="_blank"><img src="https://latex.codecogs.com/png.latex?Fj&space;=&space;\begin{bmatrix}&space;\frac{\partial&space;x_{t&plus;1}}{\partial&space;x}&space;&&space;\frac{\partial&space;x_{t&plus;1}}{\partial&space;y}&space;&&space;\frac{\partial&space;x_{t&plus;1}}{\partial&space;yaw}&space;&&space;\frac{\partial&space;x_{t&plus;1}}{\partial&space;v}&space;\\&space;\frac{\partial&space;y_{t&plus;1}}{\partial&space;x}&space;&&space;\frac{\partial&space;y_{t&plus;1}}{\partial&space;y}&space;&&space;\frac{\partial&space;y_{t&plus;1}}{\partial&space;yaw}&space;&&space;\frac{\partial&space;y_{t&plus;1}}{\partial&space;v}&space;\\&space;\frac{\partial&space;yaw}{\partial&space;x}&space;&&space;\frac{\partial&space;yaw}{\partial&space;y}&space;&&space;\frac{\partial&space;yaw}{\partial&space;yaw}&space;&&space;\frac{\partial&space;yaw}{\partial&space;v}&space;\\&space;\frac{\partial&space;v}{\partial&space;x}&space;&&space;\frac{\partial&space;v}{\partial&space;y}&space;&&space;\frac{\partial&space;v}{\partial&space;yaw}&space;&&space;\frac{\partial&space;v}{\partial&space;v}&space;\\&space;\end{bmatrix}" title="Fj = \begin{bmatrix} \frac{\partial x_{t+1}}{\partial x} & \frac{\partial x_{t+1}}{\partial y} & \frac{\partial x_{t+1}}{\partial yaw} & \frac{\partial x_{t+1}}{\partial v} \\ \frac{\partial y_{t+1}}{\partial x} & \frac{\partial y_{t+1}}{\partial y} & \frac{\partial y_{t+1}}{\partial yaw} & \frac{\partial y_{t+1}}{\partial v} \\ \frac{\partial yaw}{\partial x} & \frac{\partial yaw}{\partial y} & \frac{\partial yaw}{\partial yaw} & \frac{\partial yaw}{\partial v} \\ \frac{\partial v}{\partial x} & \frac{\partial v}{\partial y} & \frac{\partial v}{\partial yaw} & \frac{\partial v}{\partial v} \\ \end{bmatrix}" /></a>

hence:

<img src="https://latex.codecogs.com/png.latex?\frac{dx}{dyaw}&space;=&space;-v*dt*sin(yaw)&space;\\&space;\frac{dx}{dv}&space;=&space;dt*cos(yaw)&space;\\&space;\frac{dy}{dyaw}&space;=&space;v*dt*cos(yaw)&space;\\&space;\frac{dy}{dv}&space;=&space;dt*sin(yaw)&space;\\" title="\frac{dx}{dyaw} = -v*dt*sin(yaw) \\ \frac{dx}{dv} = dt*cos(yaw) \\ \frac{dy}{dyaw} = v*dt*cos(yaw) \\ \frac{dy}{dv} = dt*sin(yaw) \\" />
#### (2)update
if we use GPS as a positioning observation
the measure function is 

<a href="https://www.codecogs.com/eqnedit.php?latex=z&space;=&space;\begin{bmatrix}&space;1&space;&&space;0&space;&&space;0&space;&&space;0&space;\\&space;0&space;&&space;1&space;&&space;0&space;&&space;0&space;\\&space;\end{bmatrix}&space;\begin{bmatrix}&space;x&space;\\&space;y&space;\\&space;yaw&space;\\&space;v&space;\\&space;\end{bmatrix}" target="_blank"><img src="https://latex.codecogs.com/png.latex?z&space;=&space;\begin{bmatrix}&space;1&space;&&space;0&space;&&space;0&space;&&space;0&space;\\&space;0&space;&&space;1&space;&&space;0&space;&&space;0&space;\\&space;\end{bmatrix}&space;\begin{bmatrix}&space;x&space;\\&space;y&space;\\&space;yaw&space;\\&space;v&space;\\&space;\end{bmatrix}" title="z = \begin{bmatrix} 1 & 0 & 0 & 0 \\ 0 & 1 & 0 & 0 \\ \end{bmatrix} \begin{bmatrix} x \\ y \\ yaw \\ v \\ \end{bmatrix}" /></a>
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


