
import numpy as np
import pandas as pd
import utls.TDIDT as UT
import utls.tests.succ_rate_test
import pandas
import utls.learning_algos.ID3_impl as ID3

if __name__ == '__main__':

   df = pandas.read_csv('train.csv')
   Classifer_ID3 = ID3.ID3(df,0)
   data_test = pandas.read_csv('test.csv')
   success_rate = utls.tests.succ_rate_test.test(data_test,Classifer_ID3.Classify)
   print(success_rate)




