"""
This is the template for coding problems in exercise sheet 6, task 1.

Everywhere you see "YOUR CODE", it means a playground for you :P

WARNING: do not rename variables as this will break the tests.

HINT: In this exercise we have two different parameter vectors for the linear fits of both the X- and Y-Coordinate. We collect in a PARAMETER MATRIX theta=[theta_x, theta_y]. It is possible to do the math operations with this matrix instead of (as before) just the single vectors.

Have fun!
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg

# load the data
data = np.load("exercise6_task1_data.npz")
XYm = data['XYm']
dT = 0.0159 # sampling time
N = XYm.shape[0] # number of datapoints
t = dT * np.arange(N) # array of all sampling times

## (a) Fit a 4-th order polynomial through the data using linear least-squares. Plot the data and the fit for the X- and Y-coordinate. 
# order of the polynomial - y(t) = w_0 + w_1t + w_2t**2 + w_3t**3 + w_4t**4.
order = 4

# empty array to store the parameter matrix theta for bot the X and Y coordinate 
# Vector matrix theta_a = [[x_0 x_1 x_2 x_3 x_4].T   [y_0 y_1 y_2 y_3 y_4].T
theta_a = np.zeros((order+1, 2)) 
######## YOUR CODE ########
# compute the LLS estimate
Phi          = np.column_stack((np.ones(N), t, t**2, t**3, t**4)) # = np.power.outer(t, range(order+1))
theta_a[:,0] = (np.linalg.inv(Phi.T@Phi)@Phi.T@XYm)[:,0]
theta_a[:,1] = (np.linalg.inv(Phi.T@Phi)@Phi.T@XYm)[:,1]
# print(Phi)
###########################

# compute the fitting polynomials
polyx = Phi@theta_a[:,0]
polyy = Phi@theta_a[:,1]

# Figure 1: Plot solution for both coordinates seperately
plt.figure(1)
plt.subplot(2,1,1)
plt.plot(t, XYm[:,0], "o", markerfacecolor="none")
plt.plot(t, polyx)

plt.subplot(2,1,2)
plt.plot(t, XYm[:,1], "o", markerfacecolor="none")
plt.plot(t, polyy)

# Figure 2: Plot the data and LLS solution in both coordinates
plt.figure(2)
plt.plot(XYm[:,0], XYm[:,1], "ok", markerfacecolor="none")
plt.plot(XYm[:,0], XYm[:,1], "-k")
plt.plot(polyx, polyy, "-b")

# (b) Implement the RLS algorithm as described in the script (Check section 5.3.1) to estimate 4-th order polynomials to fit the data. Do not use forgetting factors yet. Plot the result against the data on the same plot as the previous question.

# parameter matrix for the RLS estimates, initial value is zero
theta_b = np.zeros((order+1,2))
# small initial value for the matrix Q
Q = np.eye(order+1) * 1e-10

# regressor matrix
Phi = np.power.outer(t, range(order+1))

######## YOUR CODE ########
# compute the recursive RLS estimate
for k in range(N):
    phi       = Phi[k,:,np.newaxis]
    nextQ     = Q + phi@phi.T
    nextTheta = theta_b + np.linalg.inv(nextQ)@phi@(XYm[k, :] - phi.T@theta_b)

    theta_b = nextTheta
    Q = nextQ
##########################

# plot the RLS solution into # Figure 2
rlsFit = Phi@theta_b
plt.figure(2)
plt.plot(rlsFit[:,0], rlsFit[:,1], "-g")
plt.legend([
    "Robot location samples", "Interpolated Robot location",
    "4th order polynomial fit", "4th order polynomial fit, recursive implementation"
])


## (c) Add a forgetting factor alpha to your algorithm and try different values for \alpha.  Plot the results against the data.
# Figure 3: RLS with different alphas
plt.figure(3)
plt.plot(XYm[:,0], XYm[:,1], "ok", markerfacecolor="none", label="robot location samples")

alphas = [0.7, 0.9]

# empty array to store the parameter matrices
# f.e theta_c[:,0,0] containt the parameter vector of size (order+1) for the x-coordinates, for the first alpha
theta_c = np.zeros((order+1,2,len(alphas)))
# print(theta_c) 
# regressor matrix
Phi = np.power.outer(t, range(order+1))

for n in range(len(alphas)):
    # small initial value for the matrix Q
    Q = np.eye(order+1) * 1e-10

    ######## YOUR CODE ########
    # compute the recursive RLS estimate
    for k in range(N):
        phi       = Phi[k,:,np.newaxis]  # Without np.newaxis -> Shape (5,)  - With np.newaxis -> Shape (5, 1)
        # Avoiding flatten the vector.
        nextQ     = alphas[n]*Q + phi@phi.T
        nextTheta = theta_c[:,:,n] + np.linalg.inv(nextQ)@phi@(XYm[k, :] - phi.T@theta_c[:,:,n])
        
        theta_c[:,:,n] = nextTheta
        Q = nextQ
    ##########################

    # plot the fitted curve
    rlsFit = Phi@theta_c[:,:,n]
    plt.plot(rlsFit[:,0], rlsFit[:,1], label=r"fit with $\alpha = %.3f$" % alphas[n])

plt.xlim([-5, 5])
plt.ylim([-5, 5])
plt.legend()


## (d) Compute the one-step-ahead prediction at each point (i.e. extrapolate your polynomial fit to the next time step). We also provided code to plot the 1-sigma confidence ellipsoid around this point, and the data.
plt.figure(4)

# regressor matrix
Phi = np.power.outer(t, range(order+1))

# two initial values for the matrix Q for each coordinate
Qx = np.eye(order+1) * 1e-10
Qy = Qx.copy()

# two different forgetting factors each coordinate
alphax = 0.85
alphay = 0.78

# array to store the one-step-ahead predictions
predictions = np.zeros((N, 2))

# iterates where the one-step-ahead prediction is plotted
ks = [3, 20, 50, 75]
k_iter = iter(range(7)) # to keep track of the current subplot

# empty array to store the parameter matrix
theta_e = np.zeros((order+1, 2))

# empty array to store the next parameter matrix
nextTheta = np.zeros_like(theta_e)

# compute the one-step-ahead prediction for each step
for k in range(N):

    ########### YOUR CODE #############
    # compute the recursive RLS update for each coordinate seperately
    phi             = Phi[k, np.newaxis]    # Phi (1,5) row vector.
    nextQx          = alphax * Qx + phi.T@phi
    nextQy          = alphay * Qy + phi.T@phi

    nextTheta[:,0]  = theta_e[:,0] + np.linalg.inv(nextQx)@phi.T@(XYm[k, 0] - phi@theta_e[:,0])
    nextTheta[:,1]  = theta_e[:,1] + np.linalg.inv(nextQy)@phi.T@(XYm[k, 1] - phi@theta_e[:,1])

    # compute the one-step-ahead prediction
    nextPoint = (phi @ theta_e)[0]
    
    #################################
    # save the prediction
    predictions[k,:] = nextPoint

    # compute the fit for this iterate
    rlsFit = Phi[0:k, :] @ theta_e
    
    # transfer values to next iteration
    theta_e = nextTheta
    Qx = nextQx
    Qy = nextQy
    
    # plot some of the predictions
    if k in ks:
        # create a nice plot of the prediction + confidence elipsoid
        plt.subplot(len(ks)//2, 2, next(k_iter)+1) 
        plt.plot(XYm[:,0], XYm[:,1], "ok")  
        plt.plot(rlsFit[:,0], rlsFit[:,1], "-")  
        plt.plot(XYm[0:k, 0], XYm[0:k, 1], "xk")
        plt.plot(XYm[k, 0], XYm[k, 1], "xr")
        plt.plot(nextPoint[0], nextPoint[1], "x")
        
        plt.xlim([-4, 4])
        plt.ylim([-4, 4])
        
        # covariances
        sigx = phi @ np.linalg.solve(nextQx, phi.T)
        sigy = phi @ np.linalg.solve(nextQy, phi.T)
        sig = np.block([
            [sigx, 0],
            [0, sigy],
        ])

        # Confidence ellipsoids
        sig_root = scipy.linalg.sqrtm(sig)
        xy = np.vstack((np.cos(np.linspace(0, 2*np.pi, 50)), np.sin(np.linspace(0, 2*np.pi, 50))))
        xy_ellipse = np.outer(nextPoint, np.ones(50)) + sig_root@xy

        plt.plot(xy_ellipse[0,:], xy_ellipse[1,:])
        
        plt.legend(
            ["all data", "current fit", "available data",
            "next datapoint", "prediction", "confidence ellipsoid"],
            loc="upper right"
        )
        plt.title("timestep %d" % k)

plt.show()
