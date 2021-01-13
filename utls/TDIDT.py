
import pandas
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

def test(df_test,Classify_func):
    correct = 0
    test_data = df_test
    features = test_data.columns[1::].values.tolist()
    index = 1
    features_dict = {}
    for f in features:
        features_dict[f] = index
        index += 1
    for i in range(0,len(test_data)):
        test_sample = test_data.iloc[i].tolist()
        actual_c = test_sample[0]
        predicted_c = Classify_func(test_sample , features_dict)
        if predicted_c == actual_c:
            correct+= 1
    return correct / len(test_data)


def aplly_k_validation(file_name):
    M_list= [3,13,33,73,123]
    succ_rate = [0,0,0,0,0]
    data = pandas.read_csv(file_name)
    kf = KFold(n_splits=5, shuffle=True, random_state=209418441)
    for i in range(0,5):
        result = next(kf.split(data,None))
        train = data.iloc[result[0]]
        df_test = data.iloc[result[1]]
        index = 0
        for M in M_list:
            print('testing M :',M)
            Classifer_ID3 = utls.learning_algos.ID3_impl.ID3(train,M)
            success_rate = utls.TDIDT.test(df_test, Classifer_ID3.Classify)
            print('i is',i,'M is',M , 'succ_rate' , success_rate)
            succ_rate[index] += success_rate
            index +=1
    avg_succ_rate = [i/5 for i in succ_rate]
    print(avg_succ_rate)




