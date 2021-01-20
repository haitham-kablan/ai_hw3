import utls.learning_algos.ID3_impl as ID3
import pandas
import numpy as np
import math

class KNN:
    def __init__(self, N, K, P, data):
        assert isinstance(data, pandas.DataFrame)

        self.N = N
        self.K = K
        self.data = data
        self.P = P

        n = len(data)
        self.trees = []
        self.centroid = []

        for i in range(0, N):
            data_per_tree = data.sample(math.floor(P * n))
            self.trees.append(ID3.ID3(df=data_per_tree, M=0))
            self.centroid.append(calc_avg(data_per_tree))



    def Classify(self,o):

        distance_from_sample = []
        features = self.data.columns[1::].values.tolist()

        for i in range(0,len(self.centroid)):
            distance_from_sample.append(calc_auclidian_distance(o,self.centroid[i],len(features),i))
        distance_from_sample.sort(reverse=True , key= lambda distance: distance[0])
        ans_for_best_K = []

        for i in range(0,self.K):
            ans_for_best_K.append(self.trees[distance_from_sample[i][1]].Classify(o))

        counts = [1 for c in ans_for_best_K if c == 'M']
        return 'M' if len(counts) >= len(ans_for_best_K)/2 else 'B'





def calc_avg(data):
    assert isinstance(data,pandas.DataFrame)

    np_values = data.values
    data_len = np_values.shape[0]
    features = data.columns[1::].values.tolist()
    avg_vector = np.zeros(len(features) + 1)
    avg_vector[0] = 0
    for i in range(1,len(features) + 1):
        avg_vector[i] = sum([row[i] for row in np_values])/data_len

    return avg_vector

def calc_auclidian_distance(o1,o2,feature_len,index):

# +1 becuase of diangsosn , 02 doesnt have diagnosis
#TDOD
    return math.sqrt(sum([(o1[i] - o2[i])**2 for i in range(1,feature_len +1) ])) , index