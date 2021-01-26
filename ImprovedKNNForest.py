
import KNNForest as knn_expr
import utls
import pandas
import utls.learning_algos.forest as forest

if __name__ == '__main__':

    df = pandas.read_csv('train.csv')
    test = pandas.read_csv('test.csv')

    #(best_n , best_k , best_p) = knn_expr.experiment(True)
    (best_n , best_k , best_p) = (20 , 9 , 0.7)
    print(' best k , n , p are: ' , (20 , 9 , 0.7))
    max = -float('inf')
    avg = 0
    len = 10
    for i in range(0, len):
        success_rate = utls.tests.succ_rate_test.test(test, forest.KNN_forest(best_n, best_k, best_p, df , True).Classify)
        if success_rate > max:
            max = success_rate
        print(success_rate)
        avg += success_rate

    print('max is: ', max)
    print('avg success rate is: ', avg / len)
