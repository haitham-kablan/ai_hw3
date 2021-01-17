

import pandas
import utls.learning_algos.ID3_impl as ID3_imp
import utls.tests.succ_rate_test as run_test
import utls.tests.k_validation as k_validations
import matplotlib.pyplot as plt
import numpy as np




def experiment(file_name):

   '''
   :How to run it:
            go to the main and simply remove the # sign before it.
            You can check the aplly_k_validation (by clicking + Ctrl on it) , if you like also
            to check the implementation of the k_validation itself.

   :param file_name: the file name to read data from
   :return: this function will return the avergae succ rate for each M that is in the list.
   '''

   M_list = [0, 4 , 15 , 45,120]
   succ_rate = k_validations.aplly_k_validation(file_name,M_list)

   # plotting the points
   plt.plot(M_list, succ_rate, color='green', linestyle='dashed', linewidth=3,
            marker='o', markerfacecolor='blue', markersize=12)

   # setting x and y axis range
   #plt.ylim(0, 1)
   #plt.xlim(1, 8)

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


   # here we bulid ID3_classifier and we save the desicion tree in it
   # u can click it for more information.
   Classifer_ID3 = ID3_imp.ID3(df,0)


   data_test = pandas.read_csv('test.csv')

   # the test method will run through the test data and classify each data using
   # the classify function of ID3_classifer that we built before.
   success_rate = run_test.test(data_test,Classifer_ID3.Classify)

   print(success_rate)

   #turn it on if you want to see the experince
   #experiment('train.csv')




