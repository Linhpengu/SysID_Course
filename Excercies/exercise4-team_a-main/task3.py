"""
This is the template for coding tasks in exercise sheet 4.

Everywhere you see "YOUR CODE", it means a playground for you :P

WARNING: do not rename variables as this will break the tests.
=============================================================================================================================

We are having a bit more exploration with the Least Squares problem, but this time with weighting!


New functions that can be useful for this exercise:
- to generate a diagonal matrix by specifying its diagonal elements as an array: 
    <https://numpy.org/doc/stable/reference/generated/numpy.diag.html>

- to calculate the square root of a matrix:
    <https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.sqrtm.html>


Recall some useful functions that already showed up in previous exercises:
- np.column_stack(): combine vectors as column entries into a matrix
- np.linspace() / np.arange(): generate a series
- np.linalg.solve(): solve x for Ax = b
- np.mean(): calculate the mean value, remember to specify the "axis" parameter properly.
- plt.plot(): plot data as a line / scatter / etc.


Have fun!
"""


import os
import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg


# locate the data file
filepath = os.path.join(os.path.dirname(__file__), "exercise4_dataset.npz")
# load the exercise data
data = np.load(filepath)

I = data['I']        # current data
U = data['U']        # voltage data
N_e = np.size(I, 0)  # number of students/experiments
N_m = np.size(I, 1)  # number of measurements per experiment

# print(I.shape)
# print(I)
# print(U.shape)
# print(N_e)
# print(N_m)

## (a) Plot all measurements 
plt.figure(1)
for d in range(N_e):
    plt.plot(I[d,:], U[d,:], "x")
plt.xlabel(r"$I$")
plt.ylabel(r"$U$")



## (b) LLS/WLS for student 1 and plot
# YOUR CODE: compute LLS/WLS estimation and fit

Phi = np.column_stack((I[0, :], np.ones(N_m)))
# Phi_test = np.stack((I.T, np.ones(N_m)), axis=1)
# Size of I is (100,15), We want to s
# print(Phi)
# print(Phi.shape)
# # print(Phi_test.shape)
# print(Phi[10, :])

PhiT_Phi     = np.matmul(Phi.T, Phi)
inv_PhiT_Phi = np.linalg.inv(PhiT_Phi)
theta_LLS_1  = np.matmul(np.matmul(inv_PhiT_Phi, Phi.T), U[0,:])
U_LLS_1 = theta_LLS_1[1] + theta_LLS_1[0] *I[0, :]

c = 100000.
diagonal_values = 1 / (c * np.arange(1, N_m + 1))
W = np.diag(diagonal_values)
# print(W.shape)

theta_WLS_1 = np.linalg.pinv(Phi.T@W@Phi)@Phi.T@W@U[0,:]
U_WLS_1 = theta_WLS_1[1] + theta_WLS_1[0] *I[0, :]

# Plotting
plt.figure(2)
# YOUR CODE: plot in the order of data, LLS fit, WLS fit  of student 1
plt.plot(I[0,:], U[0,:], "x")
plt.plot(I[0, :], U_LLS_1,"r")
plt.plot(I[0, :], U_WLS_1,"b")

# # Format the plot
plt.legend(["data", "LLS", "WLS"], loc="upper left")
plt.xlabel(r"$I$")
plt.ylabel(r"$U$")


# ## (d) LLS/WLS for all students
thetas_LLS = np.zeros((N_e,2))
thetas_WLS = np.zeros((N_e,2))
for d in range(N_e):
    # YOUR CODE: compute LLS/WLS estimation of each experiment
    Phi = np.column_stack((I[d, :], np.ones(N_m)))
    thetas_LLS[d,:] = np.linalg.inv(Phi.T@Phi)@Phi.T@U[d,:]
    thetas_WLS[d,:] = np.linalg.pinv(Phi.T@W@Phi)@Phi.T@W@U[d,:]


# ## (e) Estimate the mean and covariance of theta
# # YOUR CODE: mean & covariance of LLS
# print(thetas_LLS.shape)  (100, 2)
theta_mean_LLS      = np.array([(1/N_e)* np.sum(thetas_LLS[:,0]), (1/N_e)* np.sum(thetas_LLS[:,1])])
thetas_LLS_centered = thetas_LLS - theta_mean_LLS
sigma_LLS           = (1/(N_e - 1))*(thetas_LLS_centered).T@(thetas_LLS_centered)

# np.diag([((1/(N_e - 1))*np.sum((thetas_LLS[:,0] - theta_mean_LLS[0])@(thetas_LLS[:,0] - theta_mean_LLS[0]).T)), 
#                      ((1/(N_e - 1))*np.sum((thetas_LLS[:,1] - theta_mean_LLS[1])@(thetas_LLS[:,1] - theta_mean_LLS[1]).T))])

# print(theta_mean_LLS.shape)
# # YOUR CODE: mean & covariance of WLS
theta_mean_WLS      = np.array([(1/N_e)* np.sum(thetas_WLS[:,0]), (1/N_e)* np.sum(thetas_WLS[:,1])])
thetas_WLS_centered = thetas_WLS - theta_mean_WLS
sigma_WLS           = (1/(N_e - 1))*thetas_WLS_centered.T@thetas_WLS_centered

# np.diag([((1/(N_e - 1))*np.sum((thetas_WLS[:,0] - theta_mean_WLS[0])@(thetas_WLS[:,0] - theta_mean_WLS[0]).T)), 
#                      ((1/(N_e - 1))*np.sum((thetas_WLS[:,1] - theta_mean_WLS[1])@(thetas_WLS[:,1] - theta_mean_WLS[1]).T))])

# print(sigma_LLS)
# print(sigma_WLS)
# print(theta_mean_LLS)
# ## (f) Plot for all students
plt.figure(3)
print(sigma_LLS)
# Plot all estimation
plt.plot(thetas_LLS[:,0], thetas_LLS[:,1], "rx")
plt.plot(thetas_WLS[:,0], thetas_WLS[:,1], "bx")

# Plot the mean for estimators 
plt.plot(theta_mean_LLS[0], theta_mean_LLS[1], "r.", markersize=10)
plt.plot(theta_mean_WLS[0], theta_mean_WLS[1], "b.", markersize=10)

# Compute the square root of the covariance matrix
# YOUR CODE: properly call the function scipy.linalg.sqrtm()
sigma_LLS_root = scipy.linalg.sqrtm(sigma_LLS)    #np.array([scipy.linalg.sqrtm(sigma_LLS[0]), scipy.linalg.sqrtm(sigma_LLS[1])])
sigma_WLS_root = scipy.linalg.sqrtm(sigma_WLS)    #np.array([scipy.linalg.sqrtm(sigma_WLS[0]), scipy.linalg.sqrtm(sigma_WLS[1])])

# Generate coordinates as 50 points on a unit circle
num_xy = 50
xy = np.vstack((
    np.cos(np.linspace(0, 2*np.pi, num_xy)),
    np.sin(np.linspace(0, 2*np.pi, num_xy)),
))

# print(xy.shape)
# print(sigma_LLS.shape)
# # Generate the points of the confidence ellipse
xy_ellipse1 = theta_mean_LLS[:,np.newaxis] + sigma_LLS_root @ xy
xy_ellipse2 = theta_mean_WLS[:,np.newaxis] + sigma_WLS_root @ xy

# # Plot the confidence ellipse
plt.plot(xy_ellipse1[0,:], xy_ellipse1[1,:], "r-")
plt.plot(xy_ellipse2[0,:], xy_ellipse2[1,:], "b-")

# # Format the plot
plt.legend([
    r"$\theta_{LLS}^{(d)}$", r"$\theta_{WLS}^{(d)}$",
    r"$\bar{\theta}_{LLS}$", r"$\bar{\theta}_{WLS}$",
    r"$\Sigma_{LLS}$", r"$\Sigma_{WLS}$",
])
plt.xlabel(r"$R_0^*$")
plt.ylabel(r"$E_0^*$")


## Show all plots
# ===============
# If you don't see any plots but also no errors, it is very likely because a non-interactive backend is chosen by default.
# One possible solution you can try is to manually select another backend via "matplotlib.use()" function at the beginning.
# Here gives more details: <https://matplotlib.org/stable/users/explain/backends.html>
plt.show()
