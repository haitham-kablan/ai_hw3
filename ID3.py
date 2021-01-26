

import pandas
import utls.learning_algos.ID3_impl as ID3_imp
import utls.tests.succ_rate_test as run_test
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
import numpy as np


def aplly_k_validation(file_name, M_list):
   '''
   :param file_name: the file to read the data from
   :param M_list: the list that includes the M that we would like to test them
   :return: this function will return the average success rate for each M
   I also shut down the prints inside this function , you can turn them if you want more details.
   '''

   # initiating relevant data
   succ_rate = np.zeros(len(M_list))
   data = pandas.read_csv(file_name)
   kf = KFold(n_splits=5, shuffle=True, random_state=209418441)

   # run for each split and calc the success rate for each M and add it to the
   # succ_rate list.
   rotation = 0
   for train_index, test_index in kf.split(data):

      train = data.iloc[train_index]
      test = data.iloc[test_index]

      #print('testing for rotation = ', rotation)
      index = 0
      for M in M_list:
         Classifer_ID3 = ID3_imp.ID3(train, M )
         success_rate = run_test.test(test, Classifer_ID3.Classify)
         #print('         testing M: ', M, ' , success rate is: ', success_rate)
         succ_rate[index] += success_rate
         index += 1
      rotation += 1

   # calc the average for each M and return it
   avg_succ_rate = [i / 5 for i in succ_rate]
   #print('Average success rate:')
   #for i in range(0, 5):
   #  print('     M = ', i, ' , success rate: ', avg_succ_rate[i])
   return avg_succ_rate


def experiment(file_name):

   '''
   :How to run it:
            go to the main and simply remove the # sign before it.
            You can check the aplly_k_validation function in line 11
            to check the implementation of the k_validation itself.

   :param file_name: the file name to read data from
   :return: this function will print a graph where x axis is M and y axis is the average
   success rate for this M.
   '''

   M_list = [0, 4 , 10 , 15 , 45,130,200]
   succ_rate = aplly_k_validation(file_name,M_list)

   # plotting the points
   plt.plot(M_list, succ_rate, color='green', linestyle='dashed', linewidth=3,
            marker='o', markerfacecolor='blue', markersize=12)



   # naming the x axis
   plt.xlabel('M - axis')
   # naming the y axis
   plt.ylabel('Success rate - axis')

   # giving a title to my graph
   plt.title('k validation')

   # function to show the plot
   plt.show()



if __name__ == '__main__':

   df = pandas.read_csv('train.csv')


   # here we bulid ID3_classifier and we save the desicion tree inside it
   # u can click on it for more information.
   # the tree is bulid using TDIDT function wchich is in utls.TDIDT
   Classifer_ID3 = ID3_imp.ID3(df,0)

   # part 3.2
   # the early prune is implemented in utls.TDIDT


   # part 3.4
   #Classifer_ID3_M_4 = ID3_imp.ID3(df,4)


   data_test = pandas.read_csv('test.csv')

   # the test method will run through the test data and classify each data using
   # the classify function of ID3_classifer that we built before.
   success_rate = run_test.test(data_test,Classifer_ID3.Classify)

   # part 3.4
   #success_rate_M_4 = run_test.test(data_test,Classifer_ID3_M_4.Classify)

   print(success_rate)

   # part 3.4
   #print('For M = 4 , success_rate is: ',success_rate_M_4)

   #turn it on if you want to see the experiment
   #experiment('train.csv')




