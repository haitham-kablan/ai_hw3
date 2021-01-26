
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

def experiment_allah(Ratios):

    data = pandas.read_csv('train.csv')
    kf = KFold(n_splits=5, shuffle=True, random_state=209418441)
    avg_per_index = []
    for i in Ratios:
        avg_per_index.append([0, i])

    for train_index, test_index in kf.split(data):
        train = data.iloc[train_index]
        test = data.iloc[test_index]
        index = 0
        for ratio in Ratios:
            x = loss_rate_test.loss_rate(test , ID3_impl.ID3(train,0,ratio).Classify)
            avg_per_index[index][0] += x
            print(x)
            index +=1
        print('done rotation')
    x = min(avg_per_index , key= lambda x : x[0])
    return avg_per_index[[i for i, j in enumerate(avg_per_index) if j[0] == x[0]][0]]


def experiment(M):

    data = pandas.read_csv('train.csv')
    #sorted_features = find_best_features(data)

    sorted_features = ['perimeter_worst', 'concavity_worst', 'compactness_worst', 'perimeter_mean', 'radius_mean',
                        'area_mean', 'texture_worst', 'area_worst', 'compactness_mean', 'smoothness_se', 'texture_mean',
                        'symmetry_se', 'radius_worst', 'fractal_dimension_worst', 'compactness_se', 'smoothness_mean',
                        'concave points_se', 'symmetry_mean', 'texture_se', 'fractal_dimension_se', 'radius_se',
                        'perimeter_se', 'area_se', 'concavity_se', 'symmetry_worst', 'concave points_mean',
                        'fractal_dimension_mean', 'smoothness_worst', 'concave points_worst', 'concavity_mean']

    kf = KFold(n_splits=5, shuffle=True, random_state=209418441)
    avg_per_index = []
    for i in range(0 , len(sorted_features) ):
        avg_per_index.append([ 0 , i])

    for train_index, test_index in kf.split(data):
        index = 0
        for i in range( 0 , len(sorted_features)):
            train = (data.iloc[train_index])[['diagnosis'] + sorted_features[:i+1]]
            test = (data.iloc[test_index])[['diagnosis'] + sorted_features[:i+1]]
            loss_rate = loss_rate_test.loss_rate(test , ID3_impl.ID3(train,M).Classify)
            avg_per_index[index][0] += loss_rate

    return min(avg_per_index)



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
   test = pandas.read_csv('test.csv')


   #print(loss_rate_test.loss_rate(test,ID3_impl.ID3(data,0,1.5,True).Classify))
   ratios = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
   max23 = experiment_allah(ratios)
   print('score is: ' , max23[0] , 'val is: ' , max23[1])
   tree = ID3_impl.ID3(data,0,max23[1])
   print(succ_rate_test.test(test,tree.Classify))
   print(loss_rate_test.loss_rate(test,tree.Classify))
   # for rat in ratios:
   #     print(loss_rate_test.loss_rate(test,ID3_impl.ID3(data,0,rat).Classify))

   # best = experiment_allah(ratios)
   # print("best is: " , best[0] , "ratios is " , best[1])
   # print(loss_rate_test.loss_rate(test , ID3_impl.ID3(data,0,best[1]).Classify))
   #loss_rate_normal_0 = loss_rate_test.loss_rate(test,ID3_impl.ID3(data,0).Classify)
   #loss_rate_normal_4 = loss_rate_test.loss_rate(test,ID3_impl.ID3(data,4).Classify)
   # dont_delete_list = ['perimeter_worst', 'concavity_worst', 'compactness_worst', 'perimeter_mean', 'radius_mean', 'area_mean', 'texture_worst', 'area_worst', 'compactness_mean', 'smoothness_se', 'texture_mean', 'symmetry_se', 'radius_worst', 'fractal_dimension_worst', 'compactness_se', 'smoothness_mean', 'concave points_se', 'symmetry_mean', 'texture_se', 'fractal_dimension_se', 'radius_se', 'perimeter_se', 'area_se', 'concavity_se', 'symmetry_worst', 'concave points_mean', 'fractal_dimension_mean', 'smoothness_worst', 'concave points_worst', 'concavity_mean']
   # for i in range(0, len(dont_delete_list)):
   #     train = data[['diagnosis'] + dont_delete_list[:i + 1]]
   #     x = test[['diagnosis'] + dont_delete_list[:i + 1]]
   #     loss_rate = loss_rate_test.loss_rate(x, ID3_impl.ID3(train, 0).Classify)
   #     print(loss_rate)
   # i_0 = experiment(0)
   # new_loss_rate_0 = loss_rate_test.loss_rate(test[['diagnosis'] + dont_delete_list[:i_0[1]+1]],ID3_impl.ID3(data[['diagnosis'] + dont_delete_list[:i_0[1]+1]],0).Classify)
   # print('FOR M = 0 :')
   # print('  loss rate went from ' , new_loss_rate_0,' to ',new_loss_rate_0)
   # print('  improvement is: ' , loss_rate_normal_0/new_loss_rate_0)
   # i_4 = experiment(4)
   # new_loss_rate_4 = loss_rate_test.loss_rate(test[['diagnosis'] + dont_delete_list[:i_4[1] + 1]],
   #                                            ID3_impl.ID3(data[['diagnosis'] + dont_delete_list[:i_4[1] + 1]],
   #                                                         4).Classify)
   # print('FOR M = 4 :')
   # print('  loss rate went from ', loss_rate_normal_4, ' to ', new_loss_rate_4)
   # print('  improvement is: ', loss_rate_normal_4 / new_loss_rate_4)
