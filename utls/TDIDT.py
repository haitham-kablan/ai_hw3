
import pandas
import numpy as np



def Classify(tree , test_sample):

    feature, limit, subtress, c = tree

    if subtress == []:
        return c

    if test_sample[feature] >= limit:
        return Classify(subtress[0], test_sample)
    else:
        return Classify(subtress[1], test_sample)


def TDIT(E, F, DEFAULT, Select_Feature):

    assert isinstance(E, pandas.DataFrame)

    if E.empty:
        return None,None, [], DEFAULT

    c = E.diagnosis.mode()[0]
    all_c = len(E.loc[E['diagnosis'] == c].index) == len(E.index)

    if all_c:
        return None , None, [], c

    new_feature, limit = Select_Feature(E, F)

    subtrees = []
    subtrees.append(TDIT(E.loc[E[new_feature] >= limit],F,c, Select_Feature))
    subtrees.append(TDIT(E.loc[E[new_feature] < limit],F,c, Select_Feature))

    return new_feature, limit, subtrees, c
