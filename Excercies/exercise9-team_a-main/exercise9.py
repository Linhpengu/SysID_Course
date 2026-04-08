"""
This is the template for coding problems in exercise sheet 9.

Everywhere you see "YOUR CODE", it means a playground for you.

WARNING: do not rename variables as this will break the tests.

Have fun!
"""

import os
import numpy as np
import numpy.linalg as LA
import matplotlib.pyplot as plt


# locate the data file
filepath = os.path.join(os.path.dirname(__file__), "exercise9_data.npz")
# load the exercise data
data = np.load(filepath)
U = data['U']
P_measured = data['P_measured']
A_measured = data['A_measured']

# Parameters
N = 100 # number of measurements
n_x = 6 # number of states

n_p = 2 # number of states that are positions
n_v = n_p # number of states that are velocities
n_a = n_p # number of states that are accelerations

h = 0.5 # integration step

# drag cofficients
mu_1 = 0.1 
mu_2 = 10**(-2)

# initial state
x_0 = np.zeros(n_x)

# state noise
var_chi_p = 2*10**(-2)
var_chi_v = 4*10**(-3)
var_chi_a = 1*10**(-8)

# measurement noise
var_gamma_p = 16  # std of noise on position measurement 
var_gamma_a = 2.25*10**(-6)  # std of noise on acceleration measurement

# variance of initial state guess
var_0 = 10**(-5)

# (3) Write two functions that implement the prediction and update step of the Kalman filter.

## Predict and Update
def predict(x_estimate, P_estimate, A, b, W):
    ########### YOUR CODE #########
    x_predict = A@x_estimate + b
    P_predict = A@P_estimate@A.T + W
    return (x_predict, P_predict)
    ###############################

def update(y, x_predict, P_predict, C, V):
    ########### YOUR CODE #########
    P_estimate = np.linalg.inv(np.linalg.inv(P_predict) + (C.T@np.linalg.inv(V)@C))
    x_estimate = x_predict + P_estimate@C.T@np.linalg.inv(V)@(y - C@x_predict)
    return (x_estimate, P_estimate)  
    ###############################

# (4) For the given measurement and control trajectories, $y = (y_0, \ldots, y_N)$ and $u = (u_0, \ldots, u_{N-1})$, compute the state estimates $x_{[k|k]}$ and state predictions $x_{[k|k-1]}$  where we assume an estimated initial state $x_0 \sim \mathcal{N}(0, \Sigma_0)$ where $\Sigma_0 = 10^{-5} \cdot\mathbb{I}$, i.e. we assume to know the inital state almost exactly.

########### YOUR CODE #########
## State and measurement model
A = np.array([[1, 0,       h,       0,   0,  0],
              [0, 1,       0,       h,   0,  0],
              [0, 0,       1,       0,   h,  0],
              [0, 0,       0,       1,   0,  h],
              [0, 0, -mu_1*h,       0,   1,  0],
              [0, 0,       0, -mu_2*h,   0,  1]])

B = np.array([[0, 0],
              [0, 0],
              [0, 0],
              [0, 0],
              [h,-h],
              [h, h]])

# State estimates and predictions with position measurements only
C = np.array([[1,0,0,0,0,0],
              [0,1,0,0,0,0]])

# state and measurement noise matrices
V = np.diag([var_gamma_p, var_gamma_p]) 
W = np.diag([var_chi_p, var_chi_p, var_chi_v, var_chi_v, var_chi_a, var_chi_a])
##########################

## Kalman Filter
# initialize variables of the KF
x_predict = x_0
P_predict = var_0*np.eye(n_x)
p_traj = np.zeros((N, n_p))

ind_subplot = 1 # counter for the subplots
plt.figure(1)
# iterate the KF over the measurements
for i in range(N):
    
    y_tilde = P_measured[i,:] # get the current measurement
    b = B @ U[i,:] # get model into corrent form
    
    # update the of the KF
    ############# YOUR CODE ##########
    x_current, P_current = update(y_tilde, x_predict, P_predict, C, V)
    ##################################

    p_traj[i,:] = x_current[0:2] # store the predicted position
    
    # prediction step of the KF
    ############# YOUR CODE ##########
    x_predict, P_predict = predict(x_current, P_current, A, b, W)
    
    ##################################

    # create a new subplot every 30 measurements
    if (i+1) % 30 == 10:
        plt.subplot(2,4,ind_subplot )
        ind_subplot  = ind_subplot +1
        plt.title(f"GPS, t_k = {i+1}")
        plt.xlim([-9,9])
        plt.ylim([-6, 9])
        
        # (5) We already provided code to plot the estimated trajectory, the predicted position $p_{[k|k-1]}$ and the corresponding confidence ellipsoids. You only have to compute $\Sigma_{p_{[k|k-1]}}$ from $P_{[k|k-1]}$. 

        # extract the position estimate and covariance from the variables
        p_predict = x_predict[0:2]
        ############# YOUR CODE ###########
        sigma_p = P_predict[0:2, 0:2]
        ##################################

        plt.plot(p_traj[0:i,0], p_traj[0:i,1], "x-")
        plt.plot(p_predict[0], p_predict[1], "o", markersize=5)
        
        # Confidence ellipsoids
        val, vec = LA.eig(sigma_p)  # eigenvalues and eigenvectors

        xy = np.vstack((np.cos(np.linspace(0, 2*np.pi, 50)), np.sin(np.linspace(0, 2*np.pi, 50))))
        xy_ellipse = p_predict[:,np.newaxis] + vec@np.sqrt(np.diag(val))@xy
        
        plt.plot(xy_ellipse[0,:], xy_ellipse[1,:])


# (7) Repeat part 5 and 6, using now both position and acceleration measurements. This approach is generally referred to as \textit{sensor fusion}, i.e. we combine data from multiple sensors that produce measurements with different units, dimensions and accuracies, in order to obtain a more accurate state estimate.

## State estimates and predictions with position + acceleration measurements

# ############# YOUR CODE ###########
C_tilde = np.array([[1,0,0,0,0,0],
                    [0,1,0,0,0,0],
                    [0,0,0,0,1,0],
                    [0,0,0,0,0,1]])

V_tilde = np.diag([var_gamma_p, var_gamma_p, var_gamma_a, var_gamma_a])
###################################

## Kalman Filter
# initialize variables of the KF
x_predict = x_0
P_predict = var_0*np.eye(n_x)
x_current = np.zeros(n_x) # clear variable
P_current = np.zeros((n_x, n_x)) # clear variable
p_traj = np.zeros((N, n_p))

# iterate the KF over the measurements
for i in range(N):
    y_tilde = np.concatenate((P_measured[i,:], A_measured[i,:]))
    b = B @ U[i,:]
    
    ############# YOUR CODE ###########
    x_current, P_current = update(y_tilde, x_predict, P_predict, C_tilde, V_tilde)
    ###################################
    
    p_traj[i,:] = x_current[0:2]
    
    ############# YOUR CODE ###########
    x_predict, P_predict = predict(x_current, P_current, A, b, W)
    ###################################
    
    # create a new subplot every 30 measurements
    if (i+1) % 30 == 10:
        plt.subplot(2,4,ind_subplot )
        ind_subplot  = ind_subplot +1
        plt.title(f"GPS + IMU, t_k = {i+1}")
        plt.xlim([-9, 9])
        plt.ylim([-6, 9])

        #extract the position estimate and covariance from the variables
        p_predict = x_predict[0:2]
        ############# YOUR CODE ###########
        sigma_p_tilde = P_predict[0:2,0:2]
        ##################################
        
        plt.plot(p_traj[0:i,0], p_traj[0:i,1], "x-")
        plt.plot(p_predict[0], p_predict[1], "o", markersize=5)
        
        # Confidence ellipsoids
        val, vec = LA.eig(sigma_p_tilde)  # eigenvalues and eigenvectors

        xy = np.vstack((np.cos(np.linspace(0, 2*np.pi, 50)), np.sin(np.linspace(0, 2*np.pi, 50))))
        xy_ellipse = p_predict[:,np.newaxis] + vec@np.sqrt(np.diag(val))@xy
        plt.plot(xy_ellipse[0,:], xy_ellipse[1,:])

plt.show()
