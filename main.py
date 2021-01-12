
import numpy as np
import pandas as pd
import utls.TDIDT as UT
import pandas
import utls.learning_algos.ID3_impl as ID3

if __name__ == '__main__':

   tree = ID3.ID3('train.csv').tree
   x = pandas.read_csv('train.csv')
   print(x[['diagnosis']].sort_values(by='diagnosis'))




