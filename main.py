
import utls.tests.loss_rate_test as test
import utls.learning_algos.ID3_impl as ID3_impl
import pandas
import utls.learning_algos.forest as forest
import utls.learning_algos.forest as forest
import numpy as np


if __name__ == '__main__':

  df = pandas.read_csv('train.csv')
  test2 = pandas.read_csv('test.csv')
  print(df)
  train = (df)[['diagnosis' , 'radius_mean']]
  print(train)
  # a = [1,2,3,4,5]
  # print(np.argmin(a))
  # x = ID3_impl.ID3(df,0)
  # y = ID3_impl.ID3(df,4)
  # x1 = test.loss_rate(test2,x.Classify)
  # y1=test.loss_rate(test2,y.Classify)
  # print('loss rate for M = 0: ' , x1)
  # print('loss rate for M = 4: ' ,y1 )
  # print('improvement ratio is: ',x1/y1)



  # #print(df.columns[1::].values.tolist())
  # print('avg is: ' , 4.879479216435738/5)
  # print('best nkp ' , 20 , 9 , 0.4)
  # print(0.02/0.001769911504424779)
  # arr = [1,2,3,4,5]
  # print(np.argmax(arr))

