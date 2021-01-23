import pandas
import math
import numpy as np
import utls.np_array_helper_functions as utls_np



class IG_MAX:

    def __init__(self,ratio):
        """
        this class is used as the select_feature function in the id3.
        :param ratio: this is only relevant in part 4 when we want to minimize the loss
        you can check the H(self,E) function for more information (line 92).
        """
        self.ratio = ratio

    def IG_binary(self,saf,feature_index,E):
        """
        this function will calc the IG for binray feature
        :param saf: the threshold that will spilt for us the data into 2 nodes
        :param feature_index: the index of the feature that we want to calculate its IG
        :param E: the train data
        :return: the ig of this feature
        """
        H_e = self.H(E)
        E_size = E.shape[0]
        sum = 0
        bigger_than_saf , lower_than_saf = utls_np.split(E,feature_index,saf)
        for c in [0,1]:
            if c == 0:
                e_i = bigger_than_saf
            else:
                e_i = lower_than_saf
            e_i_size = e_i.shape[0]
            if e_i_size == 0:
                return float('-inf')
            h_e_i = self.H(e_i)
            sum += (e_i_size/E_size) * h_e_i
        return H_e - sum




    def IG(self,feature_index,E):
        """
        :param feature_index: the index of the feature that we want to calculate its IG
        this feature is continuous so we create k-1 binary features (such as we learn in the lectures)
        and iterate through these features and calculate for each one it's ig
        :param E: the train data
        :return: this function will return the ig of the best binary feature from the k-1 features
        with its corresponding saf(threshold)
        """
        best_sum = -float('inf')
        best_saf = -float('inf')

        f_ks = np.array([row[feature_index] for row in E])
        f_ks.sort()
        k = len(f_ks)
        new_features = range(0,k-1)
        for f in new_features:
            sum = self.IG_binary((f_ks[f] + f_ks[f+1])/2 ,feature_index,E)
            if sum > best_sum:
                best_sum = sum
                best_saf = (f_ks[f] + f_ks[f+1])/2

        return  best_sum , best_saf



    def IG_max(self,E,F):
        """
        :param E: train data
        :param F: features
        :return: this function will simply iterate through the features and return the feature with the best IG
        """
        max = -float('inf')
        saf = 0
        best_feature = 0
        #starts from 1 so we dont take diagnosis
        f_index = 1
        for i in F:
            IG_val , best_saf = self.IG(f_index,E)

            if IG_val >= max:
                max = IG_val
                best_feature = f_index
                saf = best_saf
            f_index += 1
        return best_feature,saf

    def H(self,E):
        """
        :param E: the train data
        :return: simply calculate the entropy of the given node
        """
        sum = 0
        eps = np.finfo(float).eps

        B,M = utls_np.get_all_B_all_M(E)
        for c in ['B','M']:
            size_c  = (B if c == 'B' else M).shape[0]
            size_table = E.shape[0]
            factor = 1 if c =='M' else (10 if (len(B) / len(E) >= self.ratio) else 1)
            prob_c =  (size_c * factor)/size_table
            log =  math.log(prob_c + eps,2)
            sum+= -prob_c * log

        # this factor is used for part 4 (minimizing the loss)
        # i want to make the entropy bigger for nodes with high number of B's people
        # so the algorithm will try to avoid features that spilt for us the data to
        # nodes with high number of B's , because if the node has high number of B's
        # then id3 is likely to classify it as B , and i wanted to minimize the false
        # negative errors , so i want id3 to classify M's more than B's
        # check the dry for full and boring explanation.

        return sum
