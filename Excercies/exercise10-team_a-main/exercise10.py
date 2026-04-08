"""
This is the template for coding problems in exercise sheet 9.

Everywhere you see "YOUR CODE", it means a playground for you.

WARNING: do not rename variables as this will break the tests.

Have fun!
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio

## Load data
data = sio.loadmat('ex10_data.mat')
U = data['U']                    # the inputs applied
P_measured = data['P_measured']  # the GPS measurements of the location
Kappas_measured = data['Kappas_measured']  # the measurements of the angular velocities.


## Parameters
N = 200  # number of measurements

n_x = 5  # number of states
n_u = 2  # length of input
n_p = 2  # lenght of a position measurement p
n_kappa = 2  # length of an angular velocity measurement kappa

h = 0.05  # euler step size

# variances of the state noise chi
var_chi_p     = 1e-3
var_chi_beta  = 2e-4
var_chi_v     = 1e-6
var_chi_omega = 1e-6

# variances of the measurments gamma 
var_gamma_p = 10
var_gamma_kappa = 10**(-6)

# true and wrong guess for the inital state
x_0_true  = np.array([ 0, 0, -np.pi/2, 0, 0])
x_0_wrong = np.array([-4, 2, -np.pi/2, 0, 0])

# ... and their respective variances
var_p_0 = var_gamma_p
var_x_0 = 1e-1

# ... build into a covariance matrix P_0
P_0 = np.block([
    [var_p_0*np.eye(n_p),      np.zeros((n_p, n_x-n_p))],
    [np.zeros((n_x-n_p, n_p)), var_x_0*np.eye(n_x-n_p) ],
])

# mechanical paramters of the robot   
R = 0.16
L = 0.32
m = 220
I = 9.6
Iw = 0.1

# constants
c1 = (m + 2*Iw/R**2)**(-1)
c2 = L*(I + 2*L**2*Iw/R**2)**(-1)

# plotting range
x_lims = [-5, 5]
y_lims = [-4, 12]



# Task 2: Complete the covariance matrix for the state noise chi:
W = np.block([
    [var_chi_p*np.eye(n_p), np.zeros((n_p,1)), np.zeros((n_p,1)), np.zeros((n_p,1))],
    [np.zeros((1,n_p))    , var_chi_beta     , 0                , 0                ],
    [np.zeros((1,n_p))    , 0                , var_chi_v        , 0                ],
    [np.zeros((1,n_p))    , 0                , 0                , var_chi_omega    ],
])


## Discrete time model F and its Jacobian 
#  p1 = x[0], p2 = x[1], beta=x[2], v=x[3], omega=x[4]

# Task 3: Write a function that computes F:
def F(x,u):
    """
    Computes the discrete time state dynamics by integrating the continous dynamics over the interval h using an one-step Euler-integrator.
    """ 
    ########### YOUR CODE #########
    return x + h*np.array([x[3]*np.cos(x[2]),
                           x[3]*np.sin(x[2]),
                           x[4],
                           (c1/R)*(u[0] + u[1]),
                           (c2/R)*(u[0] - u[1])])
    ###############################

def F_jacobian(x,u):
    """
    Computes the Jacobian of the discrete dynamics F with respect to the state x for a given state x and control u.
    """ 
    ########### YOUR CODE #########
    return np.array([[1, 0,-x[3]*np.sin(x[2])*h, np.cos(x[2])*h, 0],
                     [0, 1, x[3]*np.cos(x[2])*h, np.sin(x[2])*h, 0],
                     [0, 0,                   1,              0, h],
                     [0, 0,                   0,              1, 0],
                     [0, 0,                   0,              0, 1]])
    ##############################

def predict(x_estimate, P_estimate, u, F, F_jacobian, W):
    """
    The prediction function of the extended kalman filter 
    """
    ########### YOUR CODE #########
    x_predict = F(x_estimate, u)
    A = F_jacobian(x_estimate, u)
    P_predict = A@P_estimate@A.T + W
    return (x_predict, P_predict)
    ###############################

def update(y, x_predict, P_predict, C, V):
    """
    The update function of the extended kalman filter (compare ex.9)
    """
    P_estimate = np.linalg.inv(np.linalg.inv(P_predict) + C.T@np.linalg.solve(V, C))
    x_estimate = x_predict + P_estimate@C.T@np.linalg.solve(V, (y - C@x_predict))
    return (x_estimate, P_estimate)


## Measurement model 1: only p measurements
# Task 4: Specify matrix C and a covariance matrix V

########### YOUR CODE #########
C1 = np.array([[1,0,0,0,0],
               [0,1,0,0,0]])

V1 = np.diag([var_gamma_p, var_gamma_p])
###############################

## Meaurement model 2: only kappa measurements
# Task 4: Specify matrix C and a covariance matrix V
########### YOUR CODE #########
C2 = np.array([[0, 0, 0, 1/R,  L/R],
               [0, 0, 0, 1/R, -L/R]])
V2 = np.diag([var_gamma_kappa, var_gamma_kappa])
##############################

## Measurement model 3: p and kappa measurements
# Task 4: Specify matrix C and a covariance matrix V
########### YOUR CODE #########
C3 = np.vstack((C1, C2))
V3 = np.diag([var_gamma_p, var_gamma_p,var_gamma_kappa, var_gamma_kappa])
###############################

## Perform State Estimation with the EKF for different initial values and measurement models
Cs = [C1, C2, C3]
Vs = [V1, V2, V3]
x_0s = [x_0_true, x_0_wrong]
p_traj = np.zeros((2, 3, N, n_p))  # preallocation

##  Helper function
# helps to pick the right measurements (compare with Task 4)
def getMeasurement(i, k, P_measured, Kappas_measured):
    if k == 0:
        y = P_measured[i,:]
    elif k == 1:
        y = Kappas_measured[i,:]
    elif k == 2:
        y = np.concatenate((
            P_measured[i,:],
            Kappas_measured[i,:],
        ))
    else:
        raise ValueError("k must be either 0, 1 or 2")
    return y

titles = ['p measurements, $t_k$ = ', r'$\kappa$ measurements, $t_k$ = ', r'p and $\kappa$ measurements, $t_k$ = ']
fig_titles = ['True Initial State','Wrong Inital State']


for fig in range(len(fig_titles)):
    plt.figure(fig)  # create figures for each case 
    f = 1  # helper variable for subplots 
    plt.title(fig_titles[fig])  # set title for each figure
    
    for k in range(len(titles)):  # three different measurement models
        x_predict = x_0s[fig]
        P_predict = P_0
        getY = lambda i: getMeasurement(i, k, P_measured, Kappas_measured)
        
        # Task 5: Estimate the state trajectory
        for i in range(N):  #iterate over measurements
            # get current measurements y and input u from .mat file
            y = getY(i)
            u = U[i,:]
            # get C and V for current iteration
            C = Cs[k]
            V = Vs[k]

            # perform the update step:
            ########### YOUR CODE #########
            x_current, P_current = update(y, x_predict, P_predict, C, V)
            ###############################

            # save trajectory
            p_traj[fig, k, i, :] = x_current[0:2]

            # perform the prediction step
            ########### YOUR CODE #########
            x_predict, P_predict = predict(x_current, P_current, u, F, F_jacobian, W)
            ###############################

            # Plot the predictions, measurements and confidence ellipsiods
            if i % 60 == 5:
                plt.subplot(3,4,f)
                f = f+1
                plt.title(titles[k] + str(i))
                plt.xlim(x_lims)
                plt.ylim(y_lims)

                p_predict = x_predict[0:2]
                sigma_p = P_predict[0:2, 0:2]

                # plot the trajectory of the position
                plt.plot(
                    p_traj[fig, k, 0:i, 0], 
                    p_traj[fig, k, 0:i, 1],
                    '.-', markersize=8
                )
                
                # plot the predicted position for the next step 
                plt.plot(p_predict[0], p_predict[1], 'o', markersize=5)

                # Confidence ellipsoids
                val, vec = np.linalg.eig(sigma_p)  # eigenvalues and eigenvectors

                xy = np.vstack((np.cos(np.linspace(0, 2*np.pi, 50)), np.sin(np.linspace(0, 2*np.pi, 50))))
                xy_ellipse = p_predict[:,np.newaxis] + vec@np.sqrt(np.diag(val))@xy

                plt.plot(xy_ellipse[0,:], xy_ellipse[1,:])
                
                # plot the measured position
                plt.scatter(P_measured[0:i,0], P_measured[0:i,1], alpha=0.3) 

# show plots
plt.show()

## Test predict and update functions
y_test = np.ones(2)
x_test = np.ones(n_x)
u_test = np.ones(n_u)
P_test = np.eye(n_x)
C_test = np.ones((2,n_x))
V_test = np.eye(2)
u_test = np.ones(n_u)
W_test = np.eye(n_x)

x_predict_test, P_predict_test = predict(x_test, P_test, u_test, F, F_jacobian, W_test)
F_test = F(x_test,u_test)
F_jacobian_test = F_jacobian(x_test,u_test)
