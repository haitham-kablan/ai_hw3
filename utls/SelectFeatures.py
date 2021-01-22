import pandas
import math
import numpy as np
import utls.np_array_helper_functions as utls_np



class IG_MAX:
    def __init__(self,ratio):
        self.ratio = ratio
    def IG_binary(self,saf,feature_index,E):

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
        sum = 0
        eps = np.finfo(float).eps

        B,M = utls_np.get_all_B_all_M(E)
        for c in ['B','M']:
            size_c  = (B if c == 'B' else M).shape[0]
            size_table = E.shape[0]
            factor = 1 if c =='M' else (10 if (size_c/size_table > self.ratio) else 1)
            prob_c = (factor * size_c)/size_table
            log =  math.log(prob_c + eps,2)
            sum+= -prob_c * log
        return sum
