
import numpy as np

def split(array , feature_index , saf):
    bigger_than_saf = np.array([row for row in array if row[feature_index] >= saf])
    lower_than_saf = np.array([row for row in array if row[feature_index] < saf])
    return bigger_than_saf,lower_than_saf

def calc_majority(array):

    B = np.array([row for row in array if row[0] == 'B'])
    return 'B' if B.shape[0] >= array.shape[0]/2 else 'M'

def all_same_majority(array , majority):
    tmp = np.array([row for row in array if row[0] == majority])
    return tmp.shape[0] == array.shape[0]

def get_all_B_all_M(array):
    B = np.array([row for row in array if row[0] == 'B'])
    M = np.array([row for row in array if row[0] == 'M'])
    return B, M
