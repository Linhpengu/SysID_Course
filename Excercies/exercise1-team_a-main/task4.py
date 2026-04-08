"""
This is the template for coding tasks in exercise sheet 1.

Everywhere you see "YOUR CODE", it means a playground for you :P

WARNING: do not rename variables as this will break the tests.
=============================================================================================================================

Through this exercise, you will get to know the basic usage of NumPy and Matplotlib, i.e. frequently used functions/methods.
Because it is the starting point, we thought some referrences to the official documentation will ease the process.

PRECAUTION: you don't need to read through every details in the documentation, that is way too overkill. In most cases, 
grasp a rough idea of how to call a function/method with minimal arguments, and you are good to start writing your own code! 
Many typically offer some examples, which are even more intuitive. As for optional parameters, you can return to them later 
when the minimal use case cannot fulfill your demand.


We are dealing with vectors and matrices in this course, so you will need a NumPy array to store these kind of data.
Beside the classic "np.array()" function, a few functions by specifying a shape can sometimes be very handy:
    <https://numpy.org/doc/stable/reference/routines.array-creation.html#from-shape-or-value>

After you have your arrays, you can apply operators (+, -, *, /, =, ** as exponent) to them at ease. NumPy also provides 
various methods for more advanced operations, for example, the mean value:
    <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.mean.html>
Of course, you don't have to use it. There is always more than one solution to a problem.

Quite often, you want to do something with only a part of the array, such as, a single element, a specific row/column, 
a few rows/columns, etc. This is achieved by properly indexing and slicing:
    <https://numpy.org/doc/stable/user/basics.indexing.html#basic-indexing>


After the calculation, it is the exciting moment to show them off in the plots!

The "plt.figure()" function (code already written for this exercise) will create and/or switch to a plot. Then you can call 
various Pyplot functions to draw on the plot.

To draw a line:
    <https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.plot.html>

To draw a histogram:
    <https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.hist.html>

Optionally, you can also add title / xlabel / ylabel / legend to make your plot fancier.


Have fun!
"""


import os
import matplotlib.pyplot as plt
import numpy as np


Nmax = 1000 # number of samples
M = 200  # number of experiments

# locate the data file
filepath = os.path.join(os.path.dirname(__file__), "exercise1_dataset.npz")
# load exercise data
data = np.load(filepath)
u = data['u']  # Shape (1000x,200)
i = data['i']

# print(u.shape) 
# print(i.shape)

## (a)
# YOUR CODE: calculate the values
# ===============================
# "None" is just to pass the syntax check. Replace them with your code. Don't need to stick to a single line.
# (The same holds for all the following cases)
R_SA_single = np.zeros((Nmax,1))   # Shape (1000, 1) -> vector
R_LS_single = np.zeros((Nmax,1))
R_EV_single = np.zeros((Nmax,1))

true_value = np.ones((Nmax, 1)) * 2
# print(R_SA_single)
# print(R_SA_single.shape)
# Compute the SA, LS, EV of one student -> this case is student 1.
# Extract data for just the first student (index 0)
u_student1 = u[:, 0]  # Shape (1000,)
i_student1 = i[:, 0]
# print(u_student1.shape)
for k in range(0,Nmax):
    u_student1_step = u_student1[0:k+1]
    i_student1_step = i_student1[0:k+1]

    R_SA_single[k,0] = np.mean(u_student1_step/i_student1_step)
    R_LS_single[k,0] = np.sum(u_student1_step*i_student1_step)/np.sum(i_student1_step*i_student1_step)
    R_EV_single[k,0] = np.sum(u_student1_step)/np.sum(i_student1_step)

# R_SA_single_f = np.mean(u[:, 0]/i[:, 0])
# R_lS_single_f = np.dot(u[:, 0], i[:, 0]) / np.dot(i[:, 0], i[:, 0])
# R_EV_single_f = np.sum(u[:, 0])/sum(i[:, 0])

plt.figure(1)
# estimators = ['Simple Averaging (SA)', 'Least Squares (LS)', 'Error-in-Variables (EV)']
# YOUR CODE: draw on the plot 
plt.plot(R_SA_single,   label="Simple Averaging")
plt.plot(R_LS_single,   label="Least Squares")
plt.plot(R_EV_single,   label="Error-in-Variables")
# plt.plot(true_value,    color='k', linestyle='--', label="ground truth")

# # plt.bar(estimators, final_values, color=['red', 'blue', 'green'], alpha=0.7)
plt.title("Single Estimation") 
plt.ylabel("Resistance (Ohms)")
plt.legend() 

# ## (b)
# YOUR CODE: calculate the values
R_SA = np.zeros((Nmax,M))
R_LS = np.zeros((Nmax,M))
R_EV = np.zeros((Nmax,M))

for k in range(Nmax):

    u_step = u[0:k+1, :]
    i_step = i[0:k+1, :]

    R_SA[k,:] = np.mean(u_step/i_step, axis = 0)
    R_LS[k,:] = np.sum(u_step*i_step, axis = 0)/np.sum(i_step*i_step, axis = 0)
    R_EV[k,:] = np.sum(u_step, axis = 0)/np.sum(i_step, axis= 0)

plt.figure(2)
# YOUR CODE: draw on the plot 
# plt.plot(R_SA,   label="Simple Averaging")
plt.plot(R_SA, color='red', alpha=0.1)
# plt.plot(true_value, color='k', linestyle='--')

plt.title("Simple Average - 200 students") 
plt.ylabel("Resistance (Ohms)")
plt.legend() 

plt.figure(3)
# YOUR CODE: draw on the plot 
plt.plot(R_LS, color='green', alpha=0.1)
# plt.plot(true_value,  color='k', linestyle='--', label="ground truth")

plt.title("LS - 200 students") 
plt.ylabel("Resistance (Ohms)")
plt.legend() 

plt.figure(4)
# YOUR CODE: draw on the plot 

plt.plot(R_EV, color='blue', alpha=0.1)
# plt.plot(true_value, color='k', linestyle='--',    label="Ground Truth")

plt.title("EV - 200 students") 
plt.ylabel("Resistance (Ohms)")
plt.legend() 

## (c)
# YOUR CODE: calculate the values
R_SA_mean = np.mean(R_SA, axis = 1)
R_LS_mean = np.mean(R_LS, axis = 1)
R_EV_mean = np.mean(R_EV, axis = 1)

plt.figure(5)
# YOUR CODE: draw on the plot 
plt.plot(R_SA_mean,   label="Simple Averaging")
plt.plot(R_LS_mean,   label="Least Squares")
plt.plot(R_EV_mean,   label="Error-in-Variables")
# plt.plot(true_value,  color='k', linestyle='--',  label="Ground Truth")
# # plt.bar(estimators, final_values, color=['red', 'blue', 'green'], alpha=0.7)
plt.title("Single Estimation") 
plt.ylabel("Resistance (Ohms)")
plt.legend() 

## (d)
# YOUR CODE: calculate the values
# Last time step of 200 student's measurements.
R_SA_Nmax = R_SA[-1, :]   
R_LS_Nmax = R_LS[-1, :]
R_EV_Nmax = R_EV[-1, :]

plt.figure(6)
# YOUR CODE: draw on the plot 
plt.hist(R_SA_Nmax, bins=30, color='red', edgecolor='black', alpha=0.7)

#  in Matplotlib adds a vertical line spanning the entire axes height at a specified 
# -coordinate, ideal for marking thresholds or specific data points.
plt.axvline(x=2, color='k', linestyle='dashed', linewidth=2) 

plt.title("(d) Histogram of SA Estimator at N=1000")
plt.xlabel("Estimated Resistance (Ohms)")
plt.ylabel("Frequency")

plt.figure(7)
# YOUR CODE: draw on the plot 
plt.hist(R_LS_Nmax, bins=30, color='green', edgecolor='black', alpha=0.7)
plt.axvline(x=2, color='k', linestyle='dashed', linewidth=2)
plt.title("(d) Histogram of LS Estimator at N=1000")
plt.xlabel("Estimated Resistance (Ohms)")
plt.ylabel("Frequency")

plt.figure(8)
# YOUR CODE: draw on the plot 
plt.hist(R_EV_Nmax, bins=30, color='blue', edgecolor='black', alpha=0.7)
plt.axvline(x=2, color='k', linestyle='dashed', linewidth=2)

plt.title("(d) Histogram of EV Estimator at N=1000")
plt.xlabel("Estimated Resistance (Ohms)")
plt.ylabel("Frequency")

# show all plots
# ===============
# If you don't see any plots but also no errors, it is very likely because a non-interactive backend is chosen by default.
# One possible solution you can try is to manually select another backend via "matplotlib.use()" function at the beginning.
# Here gives more details: <https://matplotlib.org/stable/users/explain/backends.html>
plt.show()
