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

# def prune_experiemnt(VALS,RATIO):
#
#     data = pd.read_csv('train.csv')
#
#     rotation_index = 1
#     avg = np.zeros(len(RATIO_VALS))
#     kf = KFold(n_splits=5, shuffle=True, random_state=209418441)
#
#     for train_index, test_index in kf.split(data):
#
#         prob_index = 0
#         print('testing for rotation = ', rotation_index)
#         train = data.iloc[train_index]
#         test = data.iloc[test_index]
#         rotation_index += 1
#
#         for prob in RATIO_VALS:
#             loss_rate = loss_rate_test.loss_rate(test, ID3_impl.ID3(train, M, prob).Classify)
#             avg[prob_index] += loss_rate
#             prob_index += 1
#
#         print('best avg for rotaion: ', rotation_index, ' is for ratio = ', RATIO_VALS[np.argmin(avg)],
#               ' with loss rate: ', min(avg) / rotation_index)
#     print('best ratio for M= ', M, ' is: ', RATIO_VALS[np.argmin(avg)])
#     return RATIO_VALS[np.argmin(avg)]
#
#
#
# def experiemnt(RATIO_VALS , M):
#
#     data = pd.read_csv('train.csv')
#
#     rotation_index = 1
#     avg = np.zeros(len(RATIO_VALS))
#     kf = KFold(n_splits=5, shuffle=True, random_state=209418441)
#
#     for train_index , test_index in kf.split(data):
#
#         prob_index = 0
#         print('testing for rotation = ', rotation_index)
#         train = data.iloc[train_index]
#         test = data.iloc[test_index]
#         rotation_index += 1
#
#         for prob in RATIO_VALS:
#             loss_rate = loss_rate_test.loss_rate(test , ID3_impl.ID3(train,M,prob).Classify)
#             avg[prob_index] += loss_rate
#             prob_index +=1
#
#         print('best avg for rotaion: ' , rotation_index , ' is for ratio = ' , RATIO_VALS[np.argmin(avg)] , ' with loss rate: ' , min(avg)/rotation_index)
#     print('best ratio for M= ' , M , ' is: ',RATIO_VALS[np.argmin(avg)])
#     return RATIO_VALS[np.argmin(avg)]







if __name__ == '__main__':

    data = pd.read_csv('train.csv')
    test = pd.read_csv('test.csv')


    # normal_loss_rate = loss_rate_test.loss_rate(test, ID3_impl.ID3(data, 0).Classify)
    # normal_loss_rate_with_x = loss_rate_test.loss_rate(test, ID3_impl.ID3(data, 0,0.45).Classify)
    # normal_loss_rate_with_x_y = loss_rate_test.loss_rate(test, ID3_impl.ID3(data, 4,0.45).Classify)
    # print('loss rate normal = ', normal_loss_rate)
    # print('loss rate normal x = ', normal_loss_rate/normal_loss_rate_with_x)
    # print('loss rate normal x = ', normal_loss_rate/normal_loss_rate_with_x_y)
    #
    # #best_prob_for_m_0 = experiemnt(RATIO_VALS , 0) 0ץ9
    # loss_rate_m_0 = loss_rate_test.loss_rate(test, ID3_impl.ID3(data, 0, 0.9).Classify)
    # print('loss rate for m = 0 and val = ', 0.9, ' is: ', loss_rate_m_0)
    # print('improvement is: ', normal_loss_rate / loss_rate_m_0)
    #
    # #best_prob_for_m_4 = experiemnt(RATIO_VALS , 4)
    # loss_rate_m_4 = loss_rate_test.loss_rate(test,ID3_impl.ID3(data,4,0.6).Classify)
    # print('loss rate for m = 4 and val = ', 0.6, ' is: ', loss_rate_m_4)
    # print('improvement is: ', normal_loss_rate/ loss_rate_m_4)

