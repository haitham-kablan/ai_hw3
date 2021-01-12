
import pandas
import numpy as np



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

    if len(E) < M:
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

def test(test_file_name,Classify_func):
    correct = 0
    test_data = pandas.read_csv(test_file_name)
    features = test_data.columns[1::].values.tolist()
    index = 1
    features_dict = {}
    for f in features:
        features_dict[f] = index
        index += 1
    for i in range(0,len(test_data)):
        test_sample = test_data.iloc[i].tolist()
        c = Classify_func(test_sample , features_dict)
        if c == test_data['diagnosis'][i]:
            correct+= 1
    return correct / len(test_data)