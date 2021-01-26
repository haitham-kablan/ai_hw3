

import utls.learning_algos.forest as forest
import utls.tests.succ_rate_test
import pandas
from sklearn.model_selection import KFold
import numpy as np


def experiment(improved):
    """
    if you want to run it simply un-comment line 59
    :return: the best N,K,P
    """

    N_list = [5 ,10 ,15]
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

                    print('testing for N= ',n,', K = ',k, 'P = ',p)
                    KNN = forest.KNN_forest(N=n, K=k, P=p, data = train , improved=improved)
                    success_rate = utls.tests.succ_rate_test.test(test,KNN.Classify)
                    avg_list[index][0] += success_rate
                    avg_list[index][1] = (n,k,p)
                    print('     rate is: ',avg_list[index][0]/rotation_index)
                    index += 1
        rotation_index +=1



    best_option = max(avg_list,key= lambda x:x[0])
    print('         ****** DONE ******')
    print('best n,k,p are : ' , best_option[1] , ' with success rate: ' , best_option[0])

    return best_option[1]


if __name__ == '__main__':

    df = pandas.read_csv('train.csv')
    test = pandas.read_csv('test.csv')


    best_n, best_k, best_p = experiment(False)
    print('best n,k,p for normal are: ' ,(best_n, best_k, best_p))

    best_n_improved, best_k_improved, best_p_improved = experiment(True)
    print('best n,k,p for improved are: ', (best_n_improved, best_k_improved, best_p_improved))

    nomral_max = 0
    improved_max = 0
    normal_avg = 0
    improved_avg = 0
    times = 10

    for i in range(0,times):
        print('i is: ' , i)
        success_rate = utls.tests.succ_rate_test.test(test , forest.KNN_forest(best_n ,best_k,best_p,df,False).Classify)
        print('normal sucess rate: ' , success_rate)
        if success_rate > nomral_max:
            nomral_max = success_rate
        normal_avg += success_rate
        success_rate = utls.tests.succ_rate_test.test(test , forest.KNN_forest(best_n_improved ,best_k_improved,best_p_improved,df,True).Classify)
        print('improved sucess rate: ', success_rate)
        if success_rate > improved_max:
            improved_max = success_rate
        improved_avg += success_rate


    print('improved max is: ' , improved_max)
    print('normal max is: ' , nomral_max)
    print('average normal success rate is: ' , normal_avg / times)
    print('imrpoved normal success rate is: ' , improved_avg / times)






