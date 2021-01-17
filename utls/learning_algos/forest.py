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
        self.centroid = np.zeros(N)

        for i in range(0, N):
            data_per_tree = data.sample(P * n)
            self.trees.append(ID3.ID3(df=data_per_tree, M=0))
            self.centroid[i] = calc_avg(data_per_tree)


    def Classify(self,o):
        distance_from_sample = []
        features = self.data.columns[1::].values.tolist()
        features_dict = {}
        index = 1
        for f in features:
            features_dict[f] = index
            index += 1
        for i in self.centroid:
            distance_from_sample.append(calc_auclidian_distance(0,self.centroid[i],len(features)))
        distance_from_sample.sort(reverse=True)
        ans_for_best_K = []
        for i in range(0,self.K):
            ans_for_best_K[i] = self.trees[distance_from_sample[i]].Classify(o,features_dict)
        counts = [1 for c in ans_for_best_K if e == 'M']
        return 'M' if len(counts) >= len(ans_for_best_K) - len(counts) else 'B'







def calc_avg(data):
    assert isinstance(data,pandas.DataFrame)
    features = data.columns[1::].values.tolist()
    avg_vector = []
    for f in features:
        avg_vector.append(data[f].mean())
    return avg_vector

def calc_auclidian_distance(o1,o2,feature_len):

    return math.sqrt(sum([(o1[i] - o2[i])**2 for i in range(0,feature_len)]))