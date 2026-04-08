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

# import (and run) student scripts
import task2 as studentScript
# import task4 as studentScript

# import reference results
refResults: dict = np.load("exercise4_task2_refSol.npz")

## helper functions ##
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
def checkVariableSimilar(varName: str, 
                        tolerance: float = 1E-8,
                        absoluteValue = False) -> bool:
    """
    Compares a variable to a reference. Works also with numpy arrays of any size.  Removes singelton dimensions in arrays. Checks the shape and values.

    variable: the variable to compare
    reference_variable: the variable to compare with
    varName: Name of the variable (to create a nice output)
    tolerance: the absolute tolerance of the comparison

    """
    # get both variables from script
    variable = studentScript.__dict__.get(varName,None)
    reference_variable = refResults.get(varName,None)
    assert reference_variable is not None, f"MAJOR BUG: Variable with name {varName} does not exist in reference results!"

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

    # absolute value?
    if absoluteValue:
        variable = np.abs(variable)
        reference_variable = np.abs(reference_variable)

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
### TESTS FOR PART C ########
#############################

def test_C1():
    checkVariableSimilar('theta_opt_reg')

def test_C2():
    checkVariableSimilar('theta_opt_pm')
