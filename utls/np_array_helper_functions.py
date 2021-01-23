
import numpy as np

"""
    this file has helper function for np arrays in order to 
    process the data , I user np array instead of data frame
    for better performances
"""
def split(array , feature_index , saf):
    """
    this function will split the given array into 2 arrays according to the given feature.
    (this is used for dynamic division)
    :param array: the array to split
    :param feature_index: the index of the feature that we want to split according to it
    :param saf: the saf for this feature
    :return: bigger_than_saf is the array containing the elements that are bigger or equal than the given saf.
             lower_than_saf is the array containing the elements that are lower than the given saf.
    """
    bigger_than_saf = np.array([row for row in array if row[feature_index] >= saf])
    lower_than_saf = np.array([row for row in array if row[feature_index] < saf])
    return bigger_than_saf,lower_than_saf

def calc_majority(array):
    """

    :param array:
    :return: the majority diagnosis of the given array
    """

    B = np.array([row for row in array if row[0] == 'B'])
    return 'B' if B.shape[0] >= array.shape[0]/2 else 'M'

def all_same_majority(array , majority):
    """

    :param array:
    :param majority: the classification that we want to check if all the elements are equal to it
    :return: true if all the elements in the array have the same diagnosis as the majority
             else false
    """
    tmp = np.array([row for row in array if row[0] == majority])
    return tmp.shape[0] == array.shape[0]

def get_all_B_all_M(array):
    """
    :param array:
    :return: two arrays:
             B_array : all the elements in the given array that have B diagnosis
             A_array : all the elements in the given array that have A diagnosis
    """
    B = np.array([row for row in array if row[0] == 'B'])
    M = np.array([row for row in array if row[0] == 'M'])
    return B, M
