
import pandas
import utls.SelectFeatures
import numpy as np
from sklearn.model_selection import KFold
import utls.learning_algos.ID3_impl
import utls.np_array_helper_functions as np_utls


def Classify(tree , test_sample ):

    feature_index, saf, subtress, c = tree

    if subtress == []:
        return c

    if test_sample[feature_index] >= saf:
        return Classify(subtress[0], test_sample)
    else:
        return Classify(subtress[1], test_sample)


def TDIT(E, F, DEFAULT, Select_Feature,M):

    if E.shape[0] < M or E.shape[0] == 0:
        return None,None, [], DEFAULT

    c = np_utls.calc_majority(E)
    all_c = np_utls.all_same_majority(E,c)

    if all_c:
        return None , None, [], c

    new_feature_index, saf = Select_Feature(E, F)

    bigger_than_saf , lower_than_saf = np_utls.split(E,new_feature_index,saf)
    subtrees = []
    subtrees.append(TDIT(bigger_than_saf,F,c, Select_Feature,M))
    subtrees.append(TDIT(lower_than_saf,F,c, Select_Feature,M))

    return new_feature_index, saf, subtrees, c










