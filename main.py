import utls.tests.k_validation
import utls.tests.loss_rate_test as test
import utls.learning_algos.ID3_impl as ID3_impl
import pandas
import utls.learning_algos.forest as forest
import utls.learning_algos.forest as forest
import numpy as np


if __name__ == '__main__':

  df = pandas.read_csv('train.csv')

  assert isinstance(df,pandas.DataFrame)
  bigger_than_saf , lower_than_saf = None,None
  values = df.values
  bigger_than_saf = [values[i][feature_index] >= saf for i in range(0,len(values))]
  lower_than_saf = [values[i][feature_index] < saf for i in range(0,len(values))]
  return bigger_than_saf,lower_than_saf

