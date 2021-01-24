

import utls.learning_algos.forest as forest
import utls.tests.succ_rate_test
import pandas
from sklearn.model_selection import KFold
import numpy as np

def experiment_improved():

    N_list = [10 , 15 , 20]
    K_list = [3 , 7 , 9]
    P_list = [0.3 , 0.4 , 0.5 ,0.6 ,0.7]
    ratios = [0.1 , 0.2,0.3,0.4,0.5]

    data = pandas.read_csv('train.csv')
    test = pandas.read_csv('test.csv')

    merged = pandas.concat([data,test])

    sum = np.zeros(len(N_list) * len(K_list) * len(P_list) * len(ratios))
    avg_list = []
    for i in range(0,len(sum)):
        avg_list.append([0 , None])

    kf = KFold(n_splits=5, shuffle=True, random_state=209418441)
    rotation_index = 1
    for train_index, test_index in kf.split(merged):
        train = merged.iloc[train_index]
        test = merged.iloc[test_index]
        index = 0
        for n in N_list:
            for k in K_list:
                for p in P_list:
                    for ratio in ratios:

                        print('testing for N= ',n,', K = ',k, 'P = ',p , ' ratio = ' , ratio)
                        KNN = forest.KNN_forest(N=n, K=k, P=p, data = train , ratio=ratio)
                        success_rate = utls.tests.succ_rate_test.test(test,KNN.Classify)
                        avg_list[index][0] += success_rate
                        avg_list[index][1] = (n,k,p,ratio)
                        print('     rate is: ',avg_list[index][0]/rotation_index)
                        index += 1
        rotation_index +=1


    best_option = max(avg_list,key= lambda x:x[0])
    print('         ****** DONE ******')
    print('best n,k,p are : ' , best_option[1] , ' with success rate: ' , best_option[0])

    return best_option[1]


def experiment():

    N_list = [10 , 15 , 20]
    K_list = [3 , 7 , 9]
    P_list = [0.3 , 0.4 , 0.5 ,0.6 ,0.7]

    data = pandas.read_csv('train.csv')
    test = pandas.read_csv('test.csv')

    merged = pandas.concat([data,test])

    sum = np.zeros(len(N_list) * len(K_list) * len(P_list))
    avg_list = []
    for i in range(0,len(sum)):
        avg_list.append([0 , None])

    kf = KFold(n_splits=5, shuffle=True, random_state=209418441)
    rotation_index = 1
    for train_index, test_index in kf.split(merged):
        train = merged.iloc[train_index]
        test = merged.iloc[test_index]
        index = 0
        for n in N_list:
            for k in K_list:
                for p in P_list:

                    print('testing for N= ',n,', K = ',k, 'P = ',p)
                    KNN = forest.KNN_forest(N=n, K=k, P=p, data = train)
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
    KNN_forest = utls.learning_algos.forest.KNN_forest(N=20,K=9,P=0.4,data=df)
    KNN_forest_3 = utls.learning_algos.forest.KNN_forest(N=20,K=9,P=0.4,data=df,ratio=0.3)
    KNN_forest_4 = utls.learning_algos.forest.KNN_forest(N=20,K=9,P=0.4,data=df,ratio=0.4)
    KNN_forest_2 = utls.learning_algos.forest.KNN_forest(N=20,K=9,P=0.4,data=df,ratio=0.2)
    test = pandas.read_csv('test.csv')
    print(utls.tests.succ_rate_test.test(test,KNN_forest.Classify))
    print(utls.tests.succ_rate_test.test(test,KNN_forest_3.Classify))
    print(utls.tests.succ_rate_test.test(test,KNN_forest_4.Classify))
    print(utls.tests.succ_rate_test.test(test,KNN_forest_2.Classify))
    #experiment()
    #experiment_improved()




