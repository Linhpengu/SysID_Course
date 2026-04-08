"""
This is the template for coding problems in exercise sheet 4, task 2.

Everywhere you see "YOUR CODE", it means a playground for you :P

WARNING: do not rename variables as this will break the tests.

Have fun!
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

## Define measurements and measurement model
y = np.arange(-3,6) # measurement vector
phi = np.array([[1/6,1/6]]) # regression vector
Phi = np.repeat(phi,9,axis=0) # regressor matrix

# Define the objective function: R^2 -> R
def objectiveFunction(theta: np.ndarray):
    """
    Returns the value of the least-squares objective function for a given theta of shape (2,1).
    """
    residual = y - Phi@theta
    return 0.5*residual.T@residual

### Evaluate the objective function over a grid
theta1_grid, theta2_grid = np.meshgrid(np.linspace(0,5,30),np.linspace(0,5,30))
theta_grid = np.stack([theta1_grid,theta2_grid],axis=2).transpose((2,0,1))
res_grid = y.reshape((9,1,1)) - np.tensordot(Phi,theta_grid,(1,0))
objective_grid = 0.5*np.sum(res_grid*res_grid,axis=0)

### Plot the objective function over the grid
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.plot_surface(theta1_grid,theta2_grid,objective_grid,cmap=cm.jet,alpha=0.5)
ax.set_xlabel("Theta_1")
ax.set_ylabel("Theta_2")
ax.set_zlabel("Objective")

# (c) Find a solution to the ill posed problem using two methods
### YOUR CODE ###
theta_opt_reg = np.linalg.inv(Phi.T@Phi + 0.2*np.identity(2))@Phi.T@y
theta_opt_pm  = np.linalg.pinv(Phi.T@Phi)@Phi.T@y

#################

# useful function to plot a point on the surface
def plotPoint(x, color):
    """
    Plots a point with height equal to the objective function defined above on the surface plot.
    Plot a red x. Does nothing if x is None.
    """
    if x is None: return
    ax.plot(x[0],x[1],objectiveFunction(x),'x', color= color)

# plot the points that solve the ill posed least squares problem
plotPoint(theta_opt_reg, "r")
plotPoint(theta_opt_pm,  "g")


print(theta_opt_reg)
print(theta_opt_pm)
# show the plot
plt.show()
