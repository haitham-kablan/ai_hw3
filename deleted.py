
import utls.TDIDT as TDIDT
import utls.np_array_helper_functions as np_utls
import math
from sklearn.model_selection import KFold
import utls.learning_algos.ID3_impl as ID3_impl
import utls.np_array_helper_functions as np_utls
import utls.TDIDT as TDIDT
import utls.learning_algos.ID3_impl as ID3_impl
import pandas as pd
import utls.tests.loss_rate_test as loss_rate_test
from sklearn.model_selection import KFold
import numpy as np
import utls.tests.succ_rate_test as succ_rate_test
import math
import pandas


class prune_ID3:
    def __init__(self, Tree, V):
        self.prune_tree = prune(Tree, V)

    def Classify(self, o):
        return TDIDT.Classify(self.prune_tree, o)


def prune(Tree, V):
    feature_index, saf, subtrees, c = Tree

    if len(subtrees) == 0:
        return Tree

    samples = np_utls.split(V, feature_index, saf)

    for i in range(0, 2):
        subtrees[i] = prune(subtrees[i], samples[i])

    err_prune = 0
    err_no_prune = 0

    # calc eorrs
    for row in V:
        actual_c = row[0]
        predicted_c = TDIDT.Classify(Tree, row)
        err_prune += Evaluate(actual_c, c)
        err_no_prune += Evaluate(actual_c, predicted_c)

    if err_prune < err_no_prune:
        Tree = None, None, [], c

    return Tree


def Evaluate(actual_c, predicted_c):
    if actual_c == predicted_c:
        return 0
    else:
        return 1 if (actual_c == 'B' and predicted_c == 'M') else 10


def prune_expr(data):
    assert isinstance(data, pd.DataFrame)
    VAL_PROB = [0.1, 0.2, 0.25, 0.3, 0.35, 0.4]
    n = len(data)
    avg = []
    for val_prop in VAL_PROB:
            avg.append([0 , val_prop])
    index = 1
    orig_data = data.copy()
    kf = KFold(n_splits=5, shuffle=True, random_state=209418441)
    for train_index, test_index in kf.split(data):
        print('testing index = ', index)
        p_index = 0
        for val_prob in VAL_PROB:
            train = data.iloc[train_index]
            test = data.iloc[test_index]
            validation_data = train.sample(math.floor(len(train) * val_prob))
            train.drop(validation_data.index.tolist())
            data = orig_data
            id3_before_prune = ID3_impl.ID3(train, 0 , 0.9)
            id3_after_prune = prune_ID3(id3_before_prune.tree, validation_data.values)
            loss_rate = loss_rate_test.loss_rate(test, id3_after_prune.Classify)
            print('     val_prob= ', val_prob, 'loss_rate is: ', loss_rate)
            avg[p_index][0] += loss_rate
            p_index += 1

    return min(avg, key=lambda x: x[0])


def experiment(data, RATIO_VALS):
    assert isinstance(data, pd.DataFrame)
    kf = KFold(n_splits=5, shuffle=True, random_state=209418441)
    avg_per_ratio = np.zeros(len(RATIO_VALS))
    out_index = 1
    for train_index, test_index in kf.split(data):
        print('         Testing for index: ', out_index)
        train = data.iloc[train_index]
        test = data.iloc[test_index]
        index = 0
        out_index += 1
        for ratio in RATIO_VALS:
            id3_tree = ID3_impl.ID3(train, 0, ratio)
            loss_rate = loss_rate_test.loss_rate(test, id3_tree.Classify)
            avg_per_ratio[index] += loss_rate
            index += 1
            print('             for ratios= ', ratio, ' current avg loss rate is: ', loss_rate / out_index)
    return min(avg_per_ratio) , RATIO_VALS[np.argmin(avg_per_ratio)]



def get_local_best_feature(rest_features , best_features_till_now):

    data = pd.read_csv('train.csv')
    kf = KFold(n_splits=5, shuffle=True, random_state=209418441)
    avg_per_feature = np.zeros(len(rest_features))
    for train_index, test_index in kf.split(data):
        f_index = 0
        for f in rest_features:
            train = (data.iloc[train_index])[['diagnosis'] + best_features_till_now + [f]]

            test = (data.iloc[test_index])[['diagnosis'] + best_features_till_now + [f]]

            id3_tree = ID3_impl.ID3(train,0)
            loss_rate = loss_rate_test.loss_rate(test,id3_tree.Classify)
            avg_per_feature[f_index] += loss_rate
            f_index += 1
    avg = [row/5 for row in avg_per_feature]
    return (min(avg) , rest_features[np.argmin(avg)])

def experiment(data , RATIO_VALS):

    kf = KFold(n_splits=5, shuffle=True, random_state=209418441)
    rotation = 1
    avg_for_4 = np.zeros(len(RATIO_VALS))
    avg_for_0 = np.zeros(len(RATIO_VALS))
    for train_index , test_index in kf.split(data):
        print('testing for rotation = ', rotation)
        rotation += 1
        train = data.iloc[train_index]
        test = data.iloc[test_index]

        nrml_loss_rate = loss_rate_test.loss_rate(test, ID3_impl.ID3(train, 0).Classify)
        nrml_loss_rate_4_m = loss_rate_test.loss_rate(test, ID3_impl.ID3(train, 4).Classify)

        prob_index = 0
        for prob in RATIO_VALS:


            print('     loss rate for 0: ', nrml_loss_rate)
            print('     loss rate for 4: ', nrml_loss_rate_4_m)
            loss_rate_i_0 = loss_rate_test.loss_rate(test, ID3_impl.ID3(train, 0, prob).Classify)
            loss_rate_i_4 = loss_rate_test.loss_rate(test, ID3_impl.ID3(train, 4, prob).Classify)
            print('         loss rate for M = 0 , i = ', prob, 'is: ', loss_rate_i_0)
            if loss_rate_i_0 ==0:
                print('         shebor is inf')
            else:
                print('             shbor: ', nrml_loss_rate / loss_rate_i_0)
            print('         loss rate for M = 4 , i = ', prob, 'is: ', loss_rate_i_4)
            if loss_rate_i_4 == 0:
                print('         shebor is inf')
            else:
                print('             shbor: ', nrml_loss_rate_4_m / loss_rate_i_4)

            avg_for_4[prob_index] += loss_rate_i_4
            avg_for_0[prob_index] += loss_rate_i_0
            prob_index +=1
        print('done current iteration')
    return avg_for_0 , avg_for_4

def find_best_features(data):

    assert isinstance(data,pd.DataFrame)
    original_features = data.columns[1::].values.tolist()
    best_features_till_now = []
    x = len(original_features)
    current_loss_rate = 1
    prev_success_rate = float('inf')

    for i in range(0,x):
        current_loss_rate , best_feature = get_local_best_feature(original_features , best_features_till_now)
        best_features_till_now.append(best_feature)
        original_features.remove(best_feature)
        print('added : ' , best_feature , 'with rate: ' , current_loss_rate)

    return best_features_till_now

if __name__ == '__main__':

   #print(find_best_features(pandas.read_csv('train.csv')))
   data = pandas.read_csv('train.csv')
   orig = data.copy()
   test = pandas.read_csv('test.csv')
   orig_test = test.copy()

   dont_delete_list = ['perimeter_worst', 'concavity_worst', 'compactness_worst', 'perimeter_mean', 'radius_mean', 'area_mean', 'texture_worst', 'area_worst', 'compactness_mean', 'smoothness_se', 'texture_mean', 'symmetry_se', 'radius_worst', 'fractal_dimension_worst', 'compactness_se', 'smoothness_mean', 'concave points_se', 'symmetry_mean', 'texture_se', 'fractal_dimension_se', 'radius_se', 'perimeter_se', 'area_se', 'concavity_se', 'symmetry_worst', 'concave points_mean', 'fractal_dimension_mean', 'smoothness_worst', 'concave points_worst', 'concavity_mean']
   print(len(dont_delete_list))
   for i in range(0,len(dont_delete_list)):

       train = data[['diagnosis'] + dont_delete_list[:i+1]]
       new_test = test[['diagnosis'] + dont_delete_list[:i+1]]
       x = loss_rate_test.loss_rate(new_test,ID3_impl.ID3(train,0).Classify)
       print()