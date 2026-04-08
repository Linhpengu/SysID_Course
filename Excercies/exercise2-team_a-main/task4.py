"""
This is the template for coding tasks in exercise sheet 2.

Everywhere you see "YOUR CODE", it means a playground for you :P

WARNING: do not rename variables as this will break the tests.
=============================================================================================================================

Through this exercise, you will deal with some parameter estimation problem, and get the feel of modelling.

There is one NumPy function that can ease your process of calculation with linear model:
    <https://numpy.org/doc/stable/reference/generated/numpy.sum.html>

There are also two other functions used in the template to deal with the 3rd-order polynomial:
    <https://numpy.org/doc/stable/reference/generated/numpy.polyfit.html>
    <https://numpy.org/doc/stable/reference/generated/numpy.polyval.html>

Have fun!
"""
## PREPARATIONS

import matplotlib.pyplot as plt
import numpy as np
import os

# the figure for plotting everything
plt.figure(1)
plt.xlim([0,100])
plt.ylim([0,50])
plt.xlabel(r"Temperature - $ T(k)$")
plt.ylabel(r"Length of steel bar - $L(k)$")

# to be used only in plotting a line
T_line = np.linspace(0, 100)

## (a)
# YOUR CODE: Plot the ∆T (k), L(k) relation using ’x’ markers.
# "None" is just to pass the syntax check. Replace them with your code. Don't need to stick to a single line.
# (The same holds for all the following cases)
# locate the data file

filepath = os.path.join(os.path.dirname(__file__), "exercise2_task4_refSol.npz")
# load exercise data
data = np.load(filepath)

# print(data.files) -> Check the data
# ['T', 'L', 'T_line', 'sum_L', 'sum_LdT', 'sum_1', 'theta_1_opt', 'theta_2_opt', 'L_1d', 'poly_coeffs', 'L_3d', 'deltaT_val', 'L_val']

# datapoints from measurement
T = np.array(data['T'])
L = np.array(data['L'])

# print(T.shape)
# print(L.shape)

# (a) Plot the data
plt.plot(T, L, marker = 'x', linestyle='None')

# (b) YOUR CODE: Calculate the experimental values of theta_1 and A:
# WARNING: do not rename variables this will break the tests!

# calculate the values
theta_1_opt = (np.mean(L)*np.sum(T) - np.sum(L*T)) / (np.mean(T)*np.sum(T) - np.sum(T*T))
theta_2_opt = np.mean(L) - theta_1_opt*np.mean(T)

# # for plotting:
L_1d = theta_1_opt * T_line + theta_2_opt
plt.plot(T_line, L_1d, "r")
# plt.plot(L_1d)

# (c) YOUR CODE: Fit third order polynomial to data
# WARNING: do not rename variables this will break the tests!
# find the poly coefficinets using polyfit
poly_coeffs = np.polyfit(T, L, 3)   #  np.polyfit(x,y, 3)  : x is the input, y is the output of model

# evaluate the polynomial with over the interval
L_3d = np.polyval(poly_coeffs, T_line)
# plot the fit

plt.plot(T_line, L_3d, "g")


# # (d) YOUR CODE: Validate fits with additional measurement
T_val = 70
L_val = 32.89

plt.plot(T_val, L_val, "x")


plt.legend(["original dataset","first order fit","third order fit","validation datapoint"])

# show all plots
# ===============
# If you don't see any plots but also no errors, it is very likely because a non-interactive backend is chosen by default.
# One possible solution you can try is to manually select another backend via "matplotlib.use()" function at the beginning.
# Here gives more details: <https://matplotlib.org/stable/users/explain/backends.html>
plt.show()
