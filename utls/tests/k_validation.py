
import pandas
from sklearn.model_selection import KFold
import utls.learning_algos.ID3_impl as ID3_imp
import utls.tests.succ_rate_test as run_test
import numpy as np

def aplly_k_validation(file_name,M_list):

    '''
    :param file_name: the file to read the data from
    :param M_list: the list that includes the M that we would like to test them
    :return: this function will return the average success rate for each M
    '''

    # initiating relevant data
    succ_rate = [0,0,0,0,0]
    data = pandas.read_csv(file_name)
    kf = KFold(n_splits=5, shuffle=True, random_state=209418441)

    # run for each split and calc the success rate for each M and add it to the
    # succ_rate list.
    exper_index = 0
    for train_index, test_index in kf.split(data):

        train = data.iloc[train_index]
        test = data.iloc[test_index]

        print('testing for i = ',exper_index)
        index = 0
        for M in M_list:
            Classifer_ID3 = ID3_imp.ID3(train,M)
            success_rate = run_test.test(test, Classifer_ID3.Classify)
            print('         testing M: ',M ,' , success rate is: ',success_rate)
            succ_rate[index] += success_rate
            index +=1
        exper_index += 1


    # calc the average for each M and return it
    avg_succ_rate = [i/5 for i in succ_rate]
    print('Average success rate:')
    for i in range(0,5):
        print('     M = ',i,' , success rate: ',avg_succ_rate[i])
    return avg_succ_rate

