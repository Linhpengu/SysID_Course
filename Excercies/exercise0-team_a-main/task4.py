"""
This is the template for coding tasks in exercise sheet 0.

Everywhere you see "YOUR CODE", it means a playground for you.

WARNING: do not rename variables as this will break the tests.
=============================================================================================================================
"""

import os
import numpy as np
import matplotlib.pyplot as plt

# locate the data file
filepath = os.path.join(os.path.dirname(__file__), "position.npy")
# load the exercise data
posData = np.load(filepath)
posDataX = posData[0, :]  # x position data is in first row
posDataY = posData[1, :]  # y position data is in second row

## (a)
plt.figure(1) # new figure
######## YOUR CODE GOES HERE ###########
# FIX: To plot a 2D trajectory or spatial data, you must pass X and Y separately.
plt.plot(posDataX, posDataY, "." ,label="Measured Path") 
plt.title("Plot Positions") 
plt.legend() 
########################################
plt.gca().axis('equal') # equal spacing


## (b)
N = posDataX.shape[0] # number of measurements
plt.figure(2)
plt.gca().axis('equal') # equal spacing

######## YOUR CODE GOES HERE ###########
posXEV = np.mean(posDataX)  # compute the average value of the x position
posYEV = np.mean(posDataY)  # compute the average value of the y position

# Optional: Plot the original data lightly in the background for context
# plt.plot(posDataX, posDataY, '.', color='lightgray', label="Raw Data")

# FIX: To plot a single coordinate point (the mean), pass both X and Y.
# You must also specify a marker style (like 'ro' for red circle, or 'kx' for black cross),
# otherwise, matplotlib tries to draw a line between a single point, resulting in nothing appearing.
plt.plot(posXEV, posYEV, 'ro', markersize=8, label="Expected Value (Mean)") 
plt.title("Mean of Position Data")
plt.legend()
########################################

plt.show() 
