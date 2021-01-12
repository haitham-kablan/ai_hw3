
import numpy as np
import pandas as pd
import utls.TDIDT as UT
import utls
import pandas
import utls.learning_algos.ID3_impl as ID3

if __name__ == '__main__':

   Classifer_ID3 = ID3.ID3('train.csv')
   utls.TDIDT.test('test.csv',Classifer_ID3.Classify)




