import numpy as np
import matplotlib.pyplot as plt

# load from file
data = np.load("exercise7_dataset.npz")
y = data["y"]

# measurements
N = y.shape[0]
noise_std = 0.3  # [m]
h = 0.25  # [s]
t_k = np.arange(1, N + 1) * h

# %% Task 2b: Fit Physical Model

# define the regressor matrix
Phi_phy = np.column_stack((np.ones(N), np.ones(N)*t_k, np.ones(N)*(-0.5*t_k**2)))

# solve the LLS problem
theta_phy = np.linalg.inv(Phi_phy.T@Phi_phy)@Phi_phy.T@y

# estimate the covariance matrox of the parameters
covmatrix_estimate = (noise_std**2 )*np.linalg.inv(Phi_phy.T@Phi_phy)
# np.linalg.inv(Phi_phy.T@Phi_phy)@Phi_phy.T@(noise_std**2*np.identity(N))@(np.linalg.inv(Phi_phy.T@Phi_phy)@Phi_phy.T).T

print(f"Physical Model estimate of g: {theta_phy[2]}")

# estimate the covariance matrix of the output
covmatrix_y = Phi_phy@covmatrix_estimate@Phi_phy.T + (noise_std**2*np.identity(N))

# obtain the standard deviation for each output
stds_y = np.sqrt(np.diag(covmatrix_y))

# print(stds_y)
# physical model, useful for plotting
def p(t: np.ndarray, theta: np.ndarray):
    """
    Evaluates the physical model at time t with parameters theta.
    :param t: time, can also be a numpy vector
    :param theta: parameters of the physical model [p0, v0, g]
    """
    return theta[0] + theta[1] * t - 1 / 2 * theta[2] * t ** 2


# plot the results
plt.figure()
plt.plot(t_k, y, 'x', label="Measurements")
plt.plot(t_k, p(t_k, theta_phy), 'C1-', label="Est. Physical Model")

# plot the 2-sigma confidence intervals with red shade
plt.fill_between(t_k, p(t_k, theta_phy) - 2 * stds_y, p(t_k, theta_phy) + 2 * stds_y, color="C1", alpha=0.2,
                 label="2-$\sigma$ Confidence Interval")

# make the plot look nice
plt.ylim([0, np.max(y) * 1.05])
plt.xlim([0, np.max(t_k) * 1.05])
plt.legend(loc='lower right')
plt.grid(alpha=0.4)
plt.xlabel("Time [s]")
plt.ylabel("Position [m]")
plt.show()

# %% Task 2e) fit ARX model
d = 2  # order of the ARX model (without offset)

# define the regressor matrix and solve the LLS problem


# from the exercise
# Phi_ARX = np.column_stack((np.ones(N-2)*y[0:N-2], np.ones(N-2)*y[1:N-1], np.ones(N-2)))
# theta_ARX = np.linalg.inv(Phi_ARX.T@Phi_ARX)@Phi_ARX.T@y[2:N]

# Extra works for science with know theta_1 = -1 and theta_2 = 2
y_ARX   = y[2:N] + y[0:N-2] - 2*y[1:N-1]
Phi_ARX = np.ones((N-2,1))

# print(Phi_ARX.shape)
theta_ARX = np.linalg.inv(Phi_ARX.T@Phi_ARX)@Phi_ARX.T@y_ARX

# compute g and print it
# g_ARX = -theta_ARX[2]/(h**2)
g_ARX = -theta_ARX/(h**2)

print(f"ARX Model estimate of g: {g_ARX}")


# function to compute the rollout of the ARX model
def ARX_rollout(theta):
    rollout = y[0:d]  # start the rollout with the first 2 measurements
    for k in range(d, N):
        # compute the prediction of the ARX model using the last two predictions

        # y_predict = np.append(rollout[k - d:k], 1).T @ theta
        y_predict = -rollout[k-2] + 2*rollout[k-1] + theta_ARX

        # Adding the filter for fun 
        gain = 0.3
        y_predict = y_predict + gain*(y[k] - y_predict)

        # append the prediction to the rollout
        rollout = np.append(rollout, y_predict)
    return rollout

# perform the rollout with th estimated parameters
y_ARX_rollout = ARX_rollout(theta_ARX)

# plot the results
plt.figure()
plt.plot(t_k, y, 'x', label="Measurements")
plt.plot(t_k, y_ARX_rollout, 'C2-', label="Est. ARX Model")

# plt.plot(t_k_ARX,y_arx_eval_phyInfo,'C3-',label="ARX Model - PhysicalInfo")
plt.ylim([0, np.max(y) * 1.05])
plt.xlim([0, np.max(t_k) * 1.05])
plt.legend()
plt.grid(alpha=0.4)
plt.xlabel("Time [s]")
plt.ylabel("Position [m]")
plt.show()
