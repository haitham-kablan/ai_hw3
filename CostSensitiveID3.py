import utls.np_array_helper_functions as np_utls
import utls.TDIDT as TDIDT
import utls.learning_algos.ID3_impl as ID3_impl
import pandas as pd
import utls.tests.loss_rate_test as loss_rate_test
from sklearn.model_selection import KFold
import numpy as np
import utls.tests.succ_rate_test as succ_rate_test
import math



def experiemnt(RATIO_VALS , M):

    data = pd.read_csv('train.csv')
    rotation_index = 1
    avg = np.zeros(len(RATIO_VALS))
    kf = KFold(n_splits=5, shuffle=True, random_state=209418441)

    for train_index , test_index in kf.split(data):

        prob_index = 0
        print('testing for rotation = ', rotation_index)
        train = data.iloc[train_index]
        test = data.iloc[test_index]
        rotation_index += 1

        for prob in RATIO_VALS:
            loss_rate = loss_rate_test.loss_rate(test , ID3_impl.ID3(train,M,prob).Classify)
            avg[prob_index] += loss_rate
            prob_index +=1

    return RATIO_VALS[np.argmin(avg)]



if __name__ == '__main__':

    data = pd.read_csv('train.csv')
    test = pd.read_csv('test.csv')

    RATIO_VALS = []
    for i in range(0,20):
        RATIO_VALS.append(i/20)

    best_prob_for_m_0 = experiemnt(RATIO_VALS , 0)
    best_prob_for_m_4 = experiemnt(RATIO_VALS , 4)

    normal_loss_rate = loss_rate_test.loss_rate(test, ID3_impl.ID3(data,0).Classify)
    loss_rate_m_0 = loss_rate_test.loss_rate(test,ID3_impl.ID3(data,0,best_prob_for_m_0))
    loss_rate_m_4 = loss_rate_test.loss_rate(test,ID3_impl.ID3(data,0,best_prob_for_m_4))

    print('loss rate normal = ' , normal_loss_rate)
    print('loss rate for m = 0 and val = ' , best_prob_for_m_0 , ' is: ' , loss_rate_m_0)
    print('improvement is: ' , normal_loss_rate , loss_rate_m_0)

    print('loss rate for m = 4 and val = ', best_prob_for_m_4, ' is: ', loss_rate_m_4)
    print('improvement is: ', normal_loss_rate, loss_rate_m_4)

