"""
This is the template for coding tasks in exercise sheet 3.

Everywhere you see "YOUR CODE", it means a playground for you :P

WARNING: do not rename variables as this will break the tests.
=============================================================================================================================

Eventually you will explore the powerful least squares estimation formulated with linear algebra!
And as usual here are two functions that might be helpful to you.

- to stack vectors as columns to form a matrix:
    <https://numpy.org/doc/stable/reference/generated/numpy.column_stack.html>

- to calculate x = A^(-1) b with specifying A, b:
    <https://numpy.org/doc/stable/reference/generated/numpy.linalg.solve.html>

- to extract the diagonal of a matrix:
    <https://numpy.org/doc/stable/reference/generated/numpy.diagonal.html>

Have fun!
"""


import os
import matplotlib.pyplot as plt
import numpy as np



## load data
filepath = os.path.join(os.path.dirname(__file__), "exercise3_task3_dataset.npz")
data = np.load(filepath)
print(data.files)
i = data['i']
u = data['u']
N = 2000
print(u.shape)

## a) Plot each dataset in a corresponding plot
plt.figure(1)
# YOUR CODE: plot dataset i, u
plt.plot(i, u, marker = 'x', linestyle='None')
plt.title("E and R Estimation") 
plt.ylabel("Voltage  (Vol)")
plt.xlabel("Currents  (Ampe)")
plt.legend() 

## b) only PAPER questions :)


## c) 
# Use the least squares estimator to find the experimental values of R and E for each of the two datasets individually.
# YOUR CODE: least squares with dataset i, u
# ============================================
# "None" is just to pass the syntax check. Replace them with your code. Don't need to stick to a single line.
# (The same holds for all the following cases)

Phi          = np.column_stack((np.ones(N), i))
PhiT_Phi     = np.matmul(Phi.T, Phi)
inv_PhiT_Phi = np.linalg.inv(PhiT_Phi)
theta_star   = np.matmul(np.matmul(inv_PhiT_Phi, Phi.T), u) # Formula found in Exercise 2 - by hand

# print(theta_star)

# Plot the linear fits through the respective measurement data
plt.figure(1)
# YOUR CODE: linear fits with dataset i, u
u_fit = theta_star[0] + theta_star[1] *i
plt.plot(i, u_fit,"r")

## d) 
# YOUR CODE: calculate the residuals (errors of the estimation)
r = u - u_fit

# plot the histograms of them
plt.figure(2)
# YOUR CODEgit
plt.hist(r, bins=30, color='blue', edgecolor='black', alpha=0.7)
# plt.axvline(x=2, color='k', linestyle='dashed', linewidth=2)

## f)
## YOUR CODE : Calculate the covariance matrix and  1-sigma standard deviation estimate
# "None" is just to pass the syntax check. Replace them with your code. Don't need to stick to a single line.
# Σˆ_Theta_hat = (||y - Phi*Theta_hat||_2^2/(N - d))*inv_PhiT_Phi

cov_mat = (np.sum(r**2)/(N - 2))*inv_PhiT_Phi   # Resuls is the 2x2 diagonal matrix [sigma^2_E     0
#                                                                                      0         sigma^2_R]

standard_deviation = np.array([np.sqrt(cov_mat[0, 0]), np.sqrt(cov_mat[1, 1])])

print(f"theta 1 estimate = {theta_star[0]} +- {standard_deviation[0]}")
print(f"theta 2 estimate = {theta_star[1]} +- {standard_deviation[1]}")

# show all plots
# ===============
# If you don't see any plots but also no errors, it is very likely because a non-interactive backend is chosen by default.
# One possible solution you can try is to manually select another backend via "matplotlib.use()" function at the beginning.
# Here gives more details: <https://matplotlib.org/stable/users/explain/backends.html>
plt.show()
