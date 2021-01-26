import utls.np_array_helper_functions as np_utls
import utls.TDIDT as TDIDT
import utls.learning_algos.ID3_impl as ID3_impl
import pandas as pd
import utls.tests.loss_rate_test as loss_rate_test
from sklearn.model_selection import KFold
import numpy as np
import utls.learning_algos.prune_id3 as pruned_id3
import utls.tests.succ_rate_test as succ_rate_test
import math

def experiment(RATIO_VLAS,M):
    """
    if you want to run it , simply un comment lines 41 - 45
    :param RATIO_VLAS: the different ratios that we want to check ,
    certain ratio describes : the minimum percentage between number of ills (diagnosed with M)
    among all the samples in order to stop recursion , as I explained in the dry.
    :param M: is the prune parameter
    :return: the best ratio (with lowest loss rate)
    """

    data = pd.read_csv('train.csv')
    kf = KFold(n_splits=5, shuffle=True, random_state=209418441)

    loss_rate_avg = []
    for val in RATIO_VLAS:
        loss_rate_avg.append( [ 0 , val] )

    for train_index, test_index in kf.split(data):
        train = data.iloc[train_index]
        test = data.iloc[test_index]
        index = 0
        for val in RATIO_VLAS:
            loss_rate_avg[index][0] += loss_rate_test.loss_rate(test,ID3_impl.ID3(train,M,True,val).Classify)
            index+=1
    return min(loss_rate_avg)[1]

if __name__ == '__main__':

    data = pd.read_csv('train.csv')
    test = pd.read_csv('test.csv')

    # RATIO_VLAS = []
    # for i in range(0,10):
    #     RATIO_VLAS.append(i/10)
    #
    # best_ratio = experiment(RATIO_VLAS,4)

    # from the experiemnt i found that best ratio is 0.7

    best_ratio = 0.7
    improved_loss_rate = loss_rate_test.loss_rate(test,ID3_impl.ID3(data,4,True,best_ratio).Classify)
    print(improved_loss_rate)




