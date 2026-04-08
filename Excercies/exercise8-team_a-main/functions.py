import numpy as np

def robot_ode(x, u, theta):
    """
    Computes the right-hand-side of the robots dynamics xdot = f(x,u,theta) for a given:

    x:      the state of shape (3,)
    u:      the applied control of shape (2,)
    theta:  vector of parameters of shape (3,)

    returns: the robot dynamics f(x,u,theta)
    """
    R_L = theta[0]
    R_R = theta[1]
    L   = theta[2]

    v = (u[0]*R_L + u[1]*R_R)/2
    beta = x[2]

    xdot = np.array([
        v * np.cos(beta),
        v * np.sin(beta),
        (u[0]*R_L - u[1]*R_R)/L,
    ])

    return xdot


def euler_step(deltaT, xk, u, theta):
    """
    Integrates the robots dynamics starting from xk using one euler step of the stepsize deltaT.

    deltaT:     integration stepsize
    xk:         start point of the integration of shape (3,)
    u:          the applied control of shape (2,)
    theta:      vector of parameters of shape (3,)
    
    returns:    the endpoint of the integration of shape (3,)
    """
    return  xk + deltaT*robot_ode(xk,u,theta)   

def sim_euler(t, x0, U, theta):
    """Computes a forward simulation of the robot dynamics by performing an an Euler integration for each interval in the time vector.
    
    t:      vector of times of the simulation of shape (N,) assuming that x(t(0)) = x_0
    x0:     initial state of the simulation of shape (3,)
    U:      matrix of stacked control inputs that are applied in each interval of shape (N-1,2)
    theta:  vector of parameters of shape (3,)

    returns: The simulated trajectory of shape (N,3)
    """
    
    # get number of integration points
    N = len(t)
    
    # compute the stepsize for each of the (N) intervals
    deltaT = np.diff(t)
    
    # create an empty matrix for the simulation result
    Xsim = np.zeros((N, len(x0)))

    # fill the first slot with the initial state x0
    Xsim[0] = x0

    # iterate the intervals and integrate the dynamics
    x_next = x0
    for i in range(N-1):
        x_next = euler_step(deltaT[i], x_next, U[i], theta)
        Xsim[i+1] = x_next

    return Xsim


def residual(theta, x0, U, t, yData, sigmaData):
    """ 
    Computes the residual vector between given measured locations y_0,y_1,...y_N and the modeled locations M_0, M_1, ... , M_N. Also incorporates the measurement variances.
    
    Since each location contains two coordinates, the function returns a vector of 2*N residuals where N is the number of measurements.

    theta:      vector of parameters of shape (3,)
    x0:         initial state of the estimation of shape (3,)
    U:          matrix of stacked control inputs that are applied in each interval of shape (N-1,2)
    t:          vector of times of the simulation of shape (N,) assuming that x(t(0)) = x_0
    yData:      matrix of measured locations of shape (N,2)
    sigmaData:  vector of measurement variances for both coordinates

    returns: a vector of residual of shape (2*N,)
    """
    ############ YOUR CODE ################

    # simulate the dynamics with the given parameters
    y_sim = sim_euler(t, x0, U, theta)

    # compute the difference
    r = yData - y_sim[:,0:2]

    # weigh the coordinates of the residuals 
    W = np.diag(1/(np.sqrt(sigmaData)))

    return (r@W).flatten()

    ########################################

