
import utls.np_array_helper_functions as np_utls
import utls.TDIDT as TDIDT
import utls.learning_algos.ID3_impl as ID3_impl
import pandas as pd
import utls.tests.loss_rate_test as loss_rate_test
from sklearn.model_selection import KFold
import numpy as np
import math

class prune_ID3:
    def __init__(self,Tree,V):
        self.prune_tree = prune(Tree,V)

    def Classify(self ,o):
        return TDIDT.Classify(self.prune_tree , o)


def prune(Tree, V):

    feature_index, saf, subtrees, c = Tree

    if len(subtrees) == 0:
        return Tree

    samples = np_utls.split(V, feature_index, saf)

    for i in range(0, 2):
        subtrees[i] = prune(subtrees[i], samples[i])

    err_prune = 0
    err_no_prune = 0

    #calc eorrs
    for row in V:
        actual_c = row[0]
        predicted_c = TDIDT.Classify(Tree,row)
        err_prune += Evaluate(actual_c , c)
        err_no_prune += Evaluate(actual_c,predicted_c)


    if err_prune < err_no_prune:
        Tree = None , None , [] , c

    return Tree


def Evaluate(actual_c,predicted_c):

    if actual_c == predicted_c:
        return 0
    else:
        return 1 if (actual_c =='B' and predicted_c == 'M' ) else 10





def experiment(data,test):

    RATIO_VALS = []
    for i in range(0,20):
        RATIO_VALS.append(4 + i/10)
    lost_rate_per_ratio = []
    for i in RATIO_VALS:
        print('building for ratio= ',i)
        id3_tree = ID3_impl.ID3(data,0,i)
        z = prune_ID3(id3_tree.tree,[])
        y = loss_rate_test.loss_rate(test,z.Classify)
        x = (loss_rate_test.loss_rate(test,id3_tree.Classify))
        lost_rate_per_ratio.append( (x, i))
        print('nrml: ' ,x)
        print('prune:',y)
        print('ssssssssssssssssss')
    print(lost_rate_per_ratio)
    return min(lost_rate_per_ratio)






if __name__ == '__main__':

    df_data = pd.read_csv('train.csv')
    test = pd.read_csv('test.csv')
    n = len(df_data)
    val , p = experiment(df_data,test)
    print('     DONE EXPERIMENT **** best ratio is for p = ' , p , ' with val: ',val)



