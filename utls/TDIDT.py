
import pandas
import utls.SelectFeatures
import numpy as np
from sklearn.model_selection import KFold
import utls.learning_algos.ID3_impl
import utls.np_array_helper_functions as np_utls
import utls.np_array_helper_functions as np_utls


def Classify(tree , test_sample ):
    """
    this function will classify the given test sample according to the given decision tree
    :param tree: the decision tree
    :param test_sample: the sample to test
    :return: the classification of the given sample
    """

    feature_index, saf, subtress, c = tree

    if subtress == []:
        return c

    if test_sample[feature_index] >= saf:
        return Classify(subtress[0], test_sample)
    else:
        return Classify(subtress[1], test_sample)


def TDIDT(E, F, DEFAULT, Select_Feature,M = 0 , improved = False , ratio = 1.5):
    """
    this function will build the TDIDT tree , excatly the same way we learned in the lectures
    if improved is true then the tree will cut its recursion if number of B in node is bigger than
    the given ratio. ( look in dry in part 4.2 for more information about improve)
    :param E: the train data
    :param F: the features
    :param DEFAULT: the default classification for the root
    :param Select_Feature: the function that will select the next feature to split the
           data according to it.
    :param M: the prune parameter (if set to 0 , then no prune will happen)
    :param improved: this flag is to check if we are on part 1 or 4 (are we trying to minimize the loss or no)
    :param ratio: the prune parameter (if set to 0 , then no prune will happen)
    :return: Tree as : Features_index , saf , subtrees , classification
             feature_index : is the index of the feature to spilt the current node by
             saf : is the saf of this feature
             subtrees : list of all children of the current node
             classification : the classification of the current node
    """

    # in case of prune
    if E.shape[0] < M or E.shape[0] == 0:
        return None,None, [], DEFAULT

    c = np_utls.calc_majority(E)
    all_c = np_utls.all_same_majority(E,c)

    if all_c:
        return None , None, [], c

    new_feature_index, saf = Select_Feature(E, F)


    # improved , part 4.2
    if improved:
        B_data,M_data = np_utls.get_all_B_all_M(E)
        if len(M_data)/len(E) > ratio:
            return None,None, [], 'M'


    bigger_than_saf , lower_than_saf = np_utls.split(E,new_feature_index,saf)
    subtrees = []

    # if we have contracting samples , then then algorithm will stuck in infinite loop
    # this is to prevent it
    if np_utls.are_2_arrays_equal(bigger_than_saf ,E) or np_utls.are_2_arrays_equal(lower_than_saf,E):
        return None, None, [], DEFAULT

    subtrees.append(TDIDT(bigger_than_saf,F,c, Select_Feature,M,improved,ratio))
    subtrees.append(TDIDT(lower_than_saf,F,c, Select_Feature,M,improved,ratio))

    return new_feature_index, saf, subtrees, c










