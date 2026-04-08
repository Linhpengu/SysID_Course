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
import task4 as studentScript

# import reference results
refResults: dict = np.load("exercise1_refSol.npz")

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


def checkValueSimilar(variable,
                      reference_variable,
                      varName,
                      tolerance: float = 1E-8,
                      absoluteValue = False,
                      customSlice: slice = None) -> bool:
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

        # apply slicing
        if customSlice is not None:
            variable = variable[customSlice]
            reference_variable = reference_variable[customSlice]

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

# check whether the variable given by its name are the same in both reference and studentscript
def checkVariableSimilar(varName: str, 
                        tolerance: float = 1E-8,
                        absoluteValue = False,
                        customSlice: slice = None) -> bool:

    # get both variables from script
    variable = studentScript.__dict__.get(varName,None)
    reference_variable = refResults.get(varName,None)
    assert reference_variable is not None, f"MAJOR BUG: Variable with name {varName} does not exist in reference results!"

    return checkValueSimilar(variable,reference_variable,varName,tolerance=tolerance,absoluteValue=absoluteValue,customSlice=customSlice)

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
### TESTS FOR PART A ########
#############################

# check if computed correctly (3 points)
def test_A_1():
    checkVariableSimilar('R_SA_single')
    # checkSimilar(studentScript.R_SA_single,refResults['R_SA_single'],'R_SA_single')
def test_A_2():
    checkVariableSimilar('R_LS_single')
    # checkSimilar(studentScript.R_LS_single,refResults['R_LS_single'],'R_LS_single')
def test_A_3():
    checkVariableSimilar('R_EV_single')

    # checkSimilar(studentScript.R_EV_single,refResults['R_EV_single'],'R_EV_single')

# check if plotted correctly (1 point)
def test_A_4():
    assert plt.fignum_exists(1), "Figure 1 does not not exist, make sure it is created with plt.figure(1)."
    fig = plt.figure(1)
    numLines = len(fig.gca().lines)
    assert numLines == 3, f"Figure 1 should show three lines, instead there are {numLines}"


#############################
### TESTS FOR PART B ########
#############################

# check if values correct and  plotted correctly (1 point)
def test_B_1():
    
    # check if values correct
    checkVariableSimilar("R_SA")
    checkVariableSimilar("R_EV")
    checkVariableSimilar("R_LS")

    # check if three figures
    for k in [2,3,4]:

        # check if created
        assert plt.fignum_exists(k), f"Figure {k} does not not exist, make sure it is created with plt.figure({k})"

        # get figures
        fig = plt.figure(k)

        # check if figure show the right number of lines
        numLines = len(fig.gca().lines)
        targetNumLines = refResults['M']
        assert  numLines == targetNumLines, f"Figure {k} should show {targetNumLines} lines, instead there are {numLines}"

#############################
### TESTS FOR PART C ########
#############################

# check if values correct and  plotted correctly (1 point)
def test_C_1():
    
    # check if values correct
    checkVariableSimilar("R_SA_mean")
    checkVariableSimilar("R_EV_mean")
    checkVariableSimilar("R_LS_mean")

    # checkSimilar(studentScript.R_SA_mean,refResults['R_SA_mean'],'R_SA_mean')
    # checkSimilar(studentScript.R_EV_mean,refResults['R_EV_mean'],'R_EV_mean')
    # checkSimilar(studentScript.R_LS_mean,refResults['R_LS_mean'],'R_LS_mean')


#############################
### TESTS FOR PART D ########
#############################

# check if values correct and  plotted correctly (1 point)
def test_D_1():
    
    # check if values correct
    checkVariableSimilar("R_SA_Nmax")
    checkVariableSimilar("R_EV_Nmax")
    checkVariableSimilar("R_LS_Nmax")

    # checkSimilar(refResults['R_SA_Nmax'],studentScript.R_SA_Nmax,'R_SA_Nmax')
    # checkSimilar(refResults['R_EV_Nmax'],studentScript.R_EV_Nmax,'R_EV_Nmax')
    # checkSimilar(refResults['R_LS_Nmax'],studentScript.R_LS_Nmax,'R_LS_Nmax')
