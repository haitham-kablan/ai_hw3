
import pandas
import utls.SelectFeatures
import numpy as np
from sklearn.model_selection import KFold
import utls.learning_algos.ID3_impl


def Classify(tree , test_sample , features_dict):

    feature, limit, subtress, c = tree

    if subtress == []:
        return c

    if test_sample[features_dict[feature]] >= limit:
        return Classify(subtress[0], test_sample , features_dict)
    else:
        return Classify(subtress[1], test_sample , features_dict)


def TDIT(E, F, DEFAULT, Select_Feature,M):

    assert isinstance(E, pandas.DataFrame)

    if len(E) < M or E.empty:
        return None,None, [], DEFAULT

    c = E.diagnosis.mode()[0]
    all_c = len(E.loc[E['diagnosis'] == c].index) == len(E.index)

    if all_c:
        return None , None, [], c

    new_feature, limit = Select_Feature(E, F)

    subtrees = []
    subtrees.append(TDIT(E.loc[E[new_feature] >= limit],F,c, Select_Feature,M))
    subtrees.append(TDIT(E.loc[E[new_feature] < limit],F,c, Select_Feature,M))

    return new_feature, limit, subtrees, c










