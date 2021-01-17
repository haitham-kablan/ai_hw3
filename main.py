import utls.tests.k_validation
import utls.tests.loss_rate_test as test
import utls.learning_algos.ID3_impl as ID3_impl
import pandas
import utls.learning_algos.forest as forest
import utls.learning_algos.forest as forest
import numpy as np


if __name__ == '__main__':

  df = pandas.read_csv('train.csv')
  forest.KNN(1,1,1,df)

