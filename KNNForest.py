

import utls.learning_algos.forest as forest
import utls.tests.succ_rate_test
import pandas
from sklearn.model_selection import KFold
import numpy as np

def experiment_improved(n,k,p):


    ratios = [0.1 , 0.2,0.3,0.4,0.5]

    data = pandas.read_csv('train.csv')

    sum = np.zeros( len(ratios))
    avg_list = []
    for i in range(0,len(sum)):
        avg_list.append([0 , None])

    kf = KFold(n_splits=5, shuffle=True, random_state=209418441)
    rotation_index = 1
    for train_index, test_index in kf.split(data):
        train = data.iloc[train_index]
        test = data.iloc[test_index]
        index = 0
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
    sum = np.zeros(len(N_list) * len(K_list) * len(P_list))
    avg_list = []
    for i in range(0,len(sum)):
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
    test = pandas.read_csv('test.csv')

    n,k,p = (20,9,0.4)
    nrml_avg = 0
    avg = 0
    x = 5
    for i in range(0,x):
        nrml_avg += utls.tests.succ_rate_test.test(test,forest.KNN_forest(N=n, K=k, P=p, data = df).Classify)
        avg += utls.tests.succ_rate_test.test(test,forest.KNN_forest(N=n, K=k, P=p, data = df ,improved=True , ratio=2).Classify)
        print('nrml abf is: ' , nrml_avg/(i+1))
        print(' abf is: ' , avg/(i+1))
        print('*******************')

    print('normal rate is : ',nrml_avg/x ,' for n,k,p = ' , ( n,k,p))
    print(' rate is : ',avg/x ,' for n,k,p,ratio = ' , (n,k,p,True))

    #experiment()
    #experiment_improved()




