
import numpy as np
import pandas as pd
import utls.TDIDT as UT
import utls
import pandas
import utls.learning_algos.ID3_impl as ID3

if __name__ == '__main__':

   df = pandas.read_csv('train.csv')
   Classifer_ID3 = ID3.ID3(df,0)
   test = pandas.read_csv('test.csv')
   success_rate = utls.TDIDT.test(test,Classifer_ID3.Classify)
   print(success_rate)




