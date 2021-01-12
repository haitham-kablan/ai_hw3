
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

def test(test_file_name,Classify):
    correct = 0
    test_data = pandas.read_csv(test_file_name)
    features = test_data.columns[1::].values.tolist()
    for i in range(0,len(test_data)):
        test_sample = {}
        for feature in features:
            test_sample[feature] = test_data[feature][i]
        c = Classify(test_sample)
        if c == test_data['diagnosis'][i]:
            correct+= 1
    print(correct / len(test_data))