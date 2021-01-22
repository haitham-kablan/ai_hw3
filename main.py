import utls.tests.k_validation
import utls.tests.loss_rate_test as test
import utls.learning_algos.ID3_impl as ID3_impl
import pandas
import utls.learning_algos.forest as forest
import utls.learning_algos.forest as forest
import numpy as np


if __name__ == '__main__':

  df = pandas.read_csv('train.csv')
  #print(df.columns[1::].values.tolist())
  print('avg is: ' , 4.879479216435738/5)
  print('best nkp ' , 20 , 9 , 0.4)
  print(0.02/0.001769911504424779)
  arr = [1,2,3,4,5]
  print(np.argmax(arr))

