

import utls.learning_algos.forest as forest
import utls.tests.succ_rate_test
import pandas
from sklearn.model_selection import KFold
import numpy as np


def experiment():
    """
    if you want to run it simply un-comment line 59
    :return: the best N,K,P
    """

    N_list = [4 , 8 ,12,15]
    K_list = [1, 3, 7 ,9]
    P_list = [0.3 , 0.4 , 0.5 ,0.6 ,0.7]

    data = pandas.read_csv('train.csv')

    avg_list = []
    for i in range(0,len(N_list) * len(K_list) * len(P_list)):
        avg_list.append([0 , None])

    kf = KFold(n_splits=5, shuffle=True, random_state=209418441)
    rotation_index = 1
    for train_index, test_index in kf.split(data):

        train = data.iloc[train_index]
        test = data.iloc[test_index]
        index = 0
        for n in N_list:
            for k in K_list:
                for p in P_list:

                    #print('testing for N= ',n,', K = ',k, 'P = ',p)
                    KNN = forest.KNN_forest(N=n, K=k, P=p, data = train , improved=False)
                    success_rate = utls.tests.succ_rate_test.test(test,KNN.Classify)
                    avg_list[index][0] += success_rate
                    avg_list[index][1] = (n,k,p)
                    #print('     rate is: ',avg_list[index][0]/rotation_index)
                    index += 1
        rotation_index +=1



    best_option = max(avg_list,key= lambda x:x[0])
    #print('         ****** DONE ******')
    #print('best n,k,p are : ' , best_option[1] , ' with success rate: ' , best_option[0])

    return best_option[1]


if __name__ == '__main__':

    df = pandas.read_csv('train.csv')
    test = pandas.read_csv('test.csv')

    #best_n, best_k, best_p = experiment()
    best_n , best_k , best_p = (12 , 9,0.7)

    max = -float('inf')
    avg = 0
    len = 10
    for i in range(0,len):
        success_rate = utls.tests.succ_rate_test.test(test , forest.KNN_forest(best_n ,best_k,best_p,df,True).Classify)
        if success_rate > max:
            max = success_rate
        avg += success_rate

    print('max is: ' , max)
    print('avg success rate is: ',avg/len)




