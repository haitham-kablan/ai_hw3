

import utls.learning_algos.forest as forest
import utls.tests.succ_rate_test
import pandas
from sklearn.model_selection import KFold
import numpy as np


def experiment(improved):
    """
    if you want to run it simply un-comment line 62
    :param: improved : is in case we want to run experiment on the improved knn forest (part 7) or not
    (part 6)
    you can also turn on the print inside this function for more information
    :return: the best N,K,P
    """

    N_list = [5 ,10 ,20]
    K_list = [3 , 7 ,9]
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
                    KNN = forest.KNN_forest(N=n, K=k, P=p, data = train , improved=improved)
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

    #best_n, best_k, best_p = experiment(False)
    best_n, best_k, best_p = 20,9,0.4

    # the implementation of the forest it self is in utls.learning_algos.forest
    success_rate = utls.tests.succ_rate_test.test(test , utls.learning_algos.forest.KNN_forest(best_n,best_k,best_p,df,False).Classify)
    print(success_rate)





