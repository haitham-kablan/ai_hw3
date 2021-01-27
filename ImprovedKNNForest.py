
import KNNForest as knn_expr
import utls
import pandas
import utls.learning_algos.forest as forest

if __name__ == '__main__':

    df = pandas.read_csv('train.csv')
    test = pandas.read_csv('test.csv')

    # best_n, best_k, best_p = experiment(True)
    best_n, best_k, best_p = 20, 9, 0.4

    # the implementation of the forest it self is in utls.learning_algos.forest
    success_rate = utls.tests.succ_rate_test.test(test,
                                                  utls.learning_algos.forest.KNN_forest(best_n, best_k, best_p, df,True).Classify)
    print(success_rate)
