
import utls.data
import numpy as np
import pandas as pd

if __name__ == '__main__':
   np.random.seed(0)
   df =  utls.data.data('train.csv').data
   print(df)


