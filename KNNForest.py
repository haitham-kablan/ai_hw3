

import utls.learning_algos.forest as forest
import utls.tests.succ_rate_test
import pandas
from sklearn.model_selection import KFold
import numpy as np

def experiment():

    N_list = [10 , 15 , 20]
    K_list = [3 , 7 , 9]
    P_list = [0.3 , 0.4 , 0.5 ,0.6 ,0.7]

    data = pandas.read_csv('train.csv')
    test = pandas.read_csv('test.csv')

    merged = pandas.concat([data,test])

    sum = np.zeros(len(N_list) * len(K_list) * len(P_list))
    kf = KFold(n_splits=5, shuffle=True, random_state=209418441)
    for train_index, test_index in kf.split(merged):
        train = merged.iloc[train_index]
        test = merged.iloc[test_index]
        index = 0
        for n in N_list:
            for k in K_list:
                for p in P_list:

                    print('testing for N= ',n,', K = ',k, 'P = ',p)
                    KNN = forest.KNN(N=n , K=k , P=p,data = train)
                    success_rate = utls.tests.succ_rate_test.test(test,KNN.Classify)
                    sum[index] += success_rate
                    print('     rate is: ',sum[index])
                    index += 1

    # todo find print sum
    print('***********************')
    print('printing results')
    index = 0
    for n in N_list:
        for k in K_list:
            for p in P_list:
                print('Avg for ',(n,k,p),'is: ',sum[index]/5)
                index+=1



if __name__ == '__main__':

    #df = pandas.read_csv('train.csv')
    #KNN = utls.learning_algos.forest.KNN(N=1,K=1,P=1,data=df)
    #test = pandas.read_csv('test.csv')
    #print(utls.tests.succ_rate_test.test(test,KNN.Classify))
    experiment()




