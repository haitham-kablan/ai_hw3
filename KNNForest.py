

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
    avg_list = []
    for i in range(0,len(sum)):
        avg_list.append([0 , None])

    kf = KFold(n_splits=5, shuffle=True, random_state=209418441)
    expr_index = 1
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
                    avg_list[index][0] += success_rate
                    avg_list[index][1] = (n,k,p)
                    print('     rate is: ',avg_list[index][0]/expr_index)
                    index += 1
        expr_index +=1

    # todo find print sum
    best_option = max(avg_list,key= lambda x:x[0])
    print('         ****** DONE ******')
    print('best n,k,p are : ' , best_option[1] , ' with success rate: ' , best_option[0])



if __name__ == '__main__':

    #df = pandas.read_csv('train.csv')
    #KNN = utls.learning_algos.forest.KNN(N=1,K=1,P=1,data=df)
    #test = pandas.read_csv('test.csv')
    #print(utls.tests.succ_rate_test.test(test,KNN.Classify))
    experiment()




