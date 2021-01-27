import utls.learning_algos.ID3_impl as ID3
import pandas
import numpy as np
import math



class KNN_forest:
    def __init__(self, N, K, P, data , improved = False ):
        """

        :param N:
        :param K:
        :param P:
        :param data:
        :param improved: whether to know we want the improved knn (part 7) or not (part 6)
        """
        assert isinstance(data, pandas.DataFrame)

        self.N = N
        self.K = K if K < N else N

        self.data = data
        self.P = P
        n = len(data)
        self.improved = improved
        self.trees = []
        self.prune_trees = []
        self.centroid = []

        for i in range(0, N):
            data_per_tree = data.sample(math.floor(P * n))
            self.trees.append(ID3.ID3(df=data_per_tree, M=0))
            self.centroid.append(calc_avg(data_per_tree))

    def Classify(self, o):

        distance_from_sample = []
        features = self.data.columns[1::].values.tolist()

        for i in range(0, len(self.centroid)):
            distance_from_sample.append(calc_auclidian_distance(o, self.centroid[i], len(features), i))
        distance_from_sample.sort(key=lambda distance: distance[0])
        ans_for_best_k = []



        for i in range(0, self.K):
            predicted_c = self.trees[distance_from_sample[i][1]].Classify(o)
            ans_for_best_k.append(predicted_c)

        M_counts = [1 for c in ans_for_best_k if c =='M']
        B_counts = [1 for c in ans_for_best_k if c =='B']


        if not self.improved:
            return 'M' if len(M_counts) > len(B_counts) else 'B'

        else:
            sum_m = 0
            sum_b = 0
            max_distance = distance_from_sample[self.K - 1][0]
            if max_distance == 0:
                max_distance = 1
            twice_max_dis = max_distance * 2

            for i in range(0,self.K):
                predicted_c = ans_for_best_k[i]
                if predicted_c == 'M':
                    sum_m += twice_max_dis - distance_from_sample[i][0]
                else:
                    sum_b += twice_max_dis - distance_from_sample[i][0]
            return 'M' if sum_m > sum_b else 'B'




def calc_avg(data):
    """

    :param data: the data to get average vector (centroid) from
    :return: the centroid of this data
    """
    assert isinstance(data, pandas.DataFrame)

    np_values = data.values
    data_len = np_values.shape[0]
    features = data.columns[1::].values.tolist()
    avg_vector = np.zeros(len(features) + 1)
    avg_vector[0] = 0
    for i in range(1, len(features) + 1):
        avg_vector[i] = sum([row[i] for row in np_values]) / data_len

    return avg_vector


def calc_auclidian_distance(o1, o2, feature_len, index):
    """

    :param o1: sample1
    :param o2: sample2 - centroid[index]
    :param feature_len: len of the features of the given data
    :param index: the index of this centroid (o2) inside self.centroid
    :return: the distance between the 2 samples with the index of this centroid inside self.centroid
    """
    return math.sqrt(sum([(o1[i] - o2[i]) ** 2 for i in range(1, feature_len + 1)])), index
