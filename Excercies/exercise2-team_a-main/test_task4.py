"""
This file contains tests that you can run to check your code.
In a terminal, navigate to this folder and run 

    pytest

to run the tests. For a more detailed output run

    pytest -v

or, to stop at the first failed test:

    pytest -x

More information can be found here: https://docs.pytest.org/en/7.1.x/reference/reference.html#command-line-flags

You are not supposed to understand or edit this file.

EDITING THIS FILE WILL NOT FIX THE PROBLEMS IN YOUR CODE!

"""

import numpy as np
import pytest

# use matplotlib backend that does not show any figures
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore", message=".*Matplotlib.*")  # ignore warnings from matplotlib (due to backend)
warnings.filterwarnings("ignore", message=".*invalid value*")  # ignore warnings from numpy (due to div by zero)
warnings.filterwarnings("ignore", message=".*Mean of empty slice*")  # ignore warnings from numpy (due to empty array)

# import (and run) student script
import task4 as studentScript

# import reference results
refResults: dict = np.load("exercise2_task4_refSol.npz")

## helper functions ##
# warning logger
def logWarning(message):
    warnings.warn(message)


# check whether two arrays are the same
def checkVariableSimilar(array: np.ndarray, reference_array: np.ndarray,varName: str) -> bool:

    # check that it is not None
    is_not_none = (array is not None)
    assert is_not_none, f"Variable {varName} is None"

    # squeeze arrays to remove singleton dimensions
    array = array.squeeze()
    reference_array = reference_array.squeeze()

    # check shapes
    # sometimes the checked arrays do have shapes like (1000,) but should have (1000,1), but thats okay
    # iterate dimensions of the two variables
    # for i in range(min([array.ndim,reference_array.ndim])):
    same_shape_as_reference = (array.shape == reference_array.shape) # to avoid long output
    assert same_shape_as_reference, f"Variable {varName} should have shape {reference_array.shape} elements in dimension but has shape {array.shape} (ignoring singleton dimensions)"

    # check values
    same_values_as_reference = np.all(np.isclose(array,reference_array,equal_nan=True)) # to avoid long output
    assert same_values_as_reference, f"Variable {varName} is not equal to its reference value" 

#############################
### TESTS FOR PART A ########
#############################

# plot the relation using 'x' markers
def test_A():
    
    # check if data entered correctly
    checkVariableSimilar(studentScript.L, refResults['L'],'L')
    checkVariableSimilar(studentScript.T, refResults['T'],'T')

    assert plt.fignum_exists(1), "Figure 1 does not not exist, make sure it is created with plt.figure(1)."
    fig = plt.figure(1)
    
    numLines = len(fig.gca().lines)
    assert numLines>0, "Figure 1 should have lines"
    
    dataLine: matplotlib.lines.Line2D = fig.gca().lines[0]
    assert np.all(np.isclose(dataLine.get_xdata(),refResults['T'],equal_nan=True)), 'The first line in the figure should plot the data'
    assert np.all(np.isclose(dataLine.get_ydata(),refResults['L'],equal_nan=True)), 'The first line in the figure should plot the data'

    assert dataLine.get_marker() == 'x', 'The data should be plotted with "x" markers'


#############################
### TESTS FOR PART B ########
#############################

# Plot the fit in the same figure as before.(0.5 points)
def test_B():
    
    
    # check if values for theta and L0 are correct
    checkVariableSimilar(studentScript.theta_1_opt, refResults['theta_1_opt'],'theta_1_opt')
    checkVariableSimilar(studentScript.theta_2_opt, refResults['theta_2_opt'],'theta_2_opt')

    assert plt.fignum_exists(1), "Figure 1 does not not exist, make sure it is created with plt.figure(1)."
    fig = plt.figure(1)

    numLines = len(fig.gca().lines)
    assert numLines>1, "The figure should have at least two lines (data, fit)"
    plotLine: matplotlib.lines.Line2D = fig.gca().lines[1]

    theta_1_opt = refResults['theta_1_opt']
    theta_2_opt = refResults['theta_2_opt']
    # check
    plotline_shows_fit = np.all(np.isclose(theta_1_opt*plotLine.get_xdata() + theta_2_opt,plotLine.get_ydata()))
    assert  plotline_shows_fit,"The second line in the figure should plot the fit"

#############################
### TESTS FOR PART C ########
#############################

# check if values correct and  plotted correctly (1 point)
def test_C():
    checkVariableSimilar(studentScript.poly_coeffs, refResults['poly_coeffs'],'poly_coeffs')

    assert plt.fignum_exists(1), "Figure 1 does not not exist, make sure it is created with plt.figure(1)."
    fig = plt.figure(1)
    numLines = len(fig.gca().lines)
    assert numLines>2, "The figure should have at least three lines (data, fit, 3rd order fit)"

#############################
### TESTS FOR PART D ########
#############################

# check if values correct and  plotted correctly (1 point)
def test_D():

    assert plt.fignum_exists(1), "Figure 1 does not not exist, make sure it is created with plt.figure(1)."
    fig = plt.figure(1)
    numLines = len(fig.gca().lines)
    assert numLines>3, "The figure should have at least four lines (data, fit, 3rd order fit, additional datapoint)"
