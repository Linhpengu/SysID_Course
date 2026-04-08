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

from matplotlib.lines import Line2D
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
import task3 as studentScript

# import reference results
refResults: dict = np.load("exercise4_task3_refSol.npz")


#############################
### HELPER FUNCTIONS ########
#############################

# warning logger
def logWarning(message):
    warnings.warn(message)


def _fixTypeIssues(variable,reference_variable):
    """
    Since the variable and it's reference can have different types but the same value,
     here we try to convert the variable. 
    """

    if type(variable) == type(reference_variable):
        return variable

    # numpy array? try to cast
    if type(reference_variable) == np.ndarray:
        # try to make original variable into numpy array
        return np.array(variable)

    # basic data type?
    if type(reference_variable) in [float, int, str]:
        # try to cast
        return type(reference_variable)(variable)

    raise TypeError(f"The reference variable has an unexpected type: {type(reference_variable)}!")


# check whether two arrays are the same
def checkVariableSimilar(variable: np.ndarray, reference_variable: np.ndarray, varName: str, tolerance: float = 1E-8) -> bool:
    """
    Compares a variable to a reference. Works also with numpy arrays of any size.  Removes singelton dimensions in arrays. Checks the shape and values.

    variable: the variable to compare
    reference_variable: the variable to compare with
    varName: Name of the variable (to create a nice output)
    tolerance: the absolute tolerance of the comparison

    """
    # type that the variable should have
    targetType = type(reference_variable)

    # check that it is not None
    is_not_none = (variable is not None)
    assert is_not_none, f"Variable {varName} is None"

    # after this, the variables should have the same type
    variable = _fixTypeIssues(variable,reference_variable)

    # numpy arrays?
    if targetType == np.ndarray:
        # squeeze arrays to remove singleton dimensions
        variable = variable.squeeze()
        reference_variable = reference_variable.squeeze()

    # check shapes
    same_shape_as_reference = (variable.shape == reference_variable.shape) # to avoid long output
    assert same_shape_as_reference, f"Variable {varName} should have shape {reference_variable.shape} elements in dimension but has shape {variable.shape} (ignoring singleton dimensions)"

    # check values
    same_values_as_reference = np.all(np.isclose(variable, reference_variable, equal_nan=True,atol=tolerance)) # to avoid long output
    assert same_values_as_reference, f"Variable {varName} with value {variable} is not equal to its reference value"


# function check whether line has correct data
def checkLineData(line: Line2D, reference_x:np.ndarray,reference_y:np.ndarray, message:str):
    """
    Checks if a matplotlib line shows the reference x and y data, if not, it prints the message.
    """
    # squeeze to remove singleton dimensions
    reference_x = reference_x.squeeze()
    reference_y = reference_y.squeeze()

    # check data
    x_data_correct = np.all(np.isclose(line.get_xdata(),reference_x,equal_nan=True))
    y_data_correct = np.all(np.isclose(line.get_ydata(),reference_y,equal_nan=True))
    
    # run assertion
    assert x_data_correct, message
    assert y_data_correct, message
 

#############################
### GENERAL TESTS ###########
#############################

@pytest.fixture(scope="session", autouse=True)
def test_general():
    # a general test to guide the students, does not give points, and should only print warningss

    # check if all names exist
    for key in refResults.keys():
        # logWarning(key)
        if not hasattr(studentScript, key):  logWarning(f"The variable {key} does not exist.")

    # check all figures for legends and labels
    # for k in plt.get_fignums():
    #     fig = plt.figure(k)
    #     if fig.gca().get_xlabel() != "":  logWarning(f"There is no x-label in figure {k}.")
    #     if fig.gca().get_ylabel() != "": logWarning(f"There is no y-label in figure {k}.") 
    #     if fig.gca().get_legend() is not None:  logWarning(f"There is no legend on figure {k}.")


#############################
### TESTS FOR PART B ########
#############################

def test_B():
    """check if values correct are plotted correctly (0.5 point)"""
    # check values
    checkVariableSimilar(studentScript.theta_LLS_1, refResults['theta_LLS_1'], "theta_LLS_1")
    checkVariableSimilar(studentScript.U_LLS_1, refResults['U_LLS_1'], "U_LLS_1")
    checkVariableSimilar(studentScript.theta_WLS_1, refResults['theta_WLS_1'], "theta_WLS_1")
    checkVariableSimilar(studentScript.U_WLS_1, refResults['U_WLS_1'], "U_WLS_1")

    # check plot exist
    assert plt.fignum_exists(2), "Figure 2 does not not exist, make sure it is created with plt.figure(2)."
    fig = plt.figure(2)

    # check only one axis object
    fig_axes = fig.get_axes()
    assert np.size(fig_axes) == 1, "Figure 1 should have only one subplot"

    # check line data & marker for line1
    [ax] = fig_axes
    line1, line2, line3 = ax.lines
    checkLineData(line1, studentScript.I[0], studentScript.U[0], "The 1st line in fig1 should be I[0]-U[0]")
    assert line1.get_marker() == 'x', 'The data should be plotted with "x" markers'
    checkLineData(line2, studentScript.I[0], refResults['U_LLS_1'], "The 2nd line in fig1 should be I[0]-U_LLS_1")
    checkLineData(line3, studentScript.I[0], refResults['U_WLS_1'], "The 3rd line in fig1 should be I[0]-U_WLS_1")


#############################
### TESTS FOR PART D ########
#############################

def test_D():
    """check if values correct (0.5 point)"""
    checkVariableSimilar(studentScript.thetas_LLS , refResults['thetas_LLS'], "thetas_LLS")
    checkVariableSimilar(studentScript.thetas_WLS , refResults['thetas_WLS'], "thetas_WLS")


#############################
### TESTS FOR PART E ########
#############################

def test_E():
    """check if values correct (0.5 point)"""
    checkVariableSimilar(studentScript.theta_mean_LLS, refResults['theta_mean_LLS'], "theta_mean_LLS")
    checkVariableSimilar(studentScript.sigma_LLS, refResults['sigma_LLS'], "sigma_LLS")
    checkVariableSimilar(studentScript.theta_mean_WLS, refResults['theta_mean_WLS'], "theta_mean_WLS")
    checkVariableSimilar(studentScript.sigma_WLS, refResults['sigma_WLS'], "sigma_WLS")


#############################
### TESTS FOR PART Fcd  ########
#############################
def test_F():
    """check if values are correct (0.5 point)"""
    checkVariableSimilar(studentScript.xy_ellipse1, refResults['xy_ellipse1'], "xy_ellipse1")
    checkVariableSimilar(studentScript.xy_ellipse2, refResults['xy_ellipse2'], "xy_ellipse2")
