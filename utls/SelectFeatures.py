import pandas
import math
import numpy as np
import utls

def split_data_frame(df,feature_index,saf,diagnosis_flag):


    assert isinstance(df,pandas.DataFrame)

    if diagnosis_flag == False:
        bigger_than_saf , lower_than_saf = None,None
        values = df.values
        bigger_than_saf = [row for row in values if row[feature_index] >= saf]
        lower_than_saf = [row for row in values if row[feature_index] < saf]
        return bigger_than_saf,lower_than_saf
    else:
        equal, other = None, None
        values = df.values
        B = [row for row in values if row[0] == 'B']
        M = [row for row in values if row[0] == 'M']
        return B, M


class IG_MAX:

    def IG_binary(self,saf,continous_feature,feature_index,E):
        assert isinstance(E,pandas.DataFrame)
        H_e = self.H(E)
        E_size = len(E.index)
        sum = 0
        bigger_than_saf , lower_than_saf = split_data_frame(E,feature_index,saf,False)
        for c in [0,1]:
            e_i = None
            if c == 0:
                e_i = bigger_than_saf
            else:
                e_i = lower_than_saf
            e_i_size = len(e_i)
            if e_i_size == 0:
                return float('-inf')
            h_e_i = self.H(e_i)
            sum += (e_i_size/E_size) * h_e_i
        return H_e - sum




    def IG(self,contuis_feature,feature_index,E):

        assert isinstance(E,pandas.DataFrame)

        best_sum = -float('inf')
        best_saf = -float('inf')

        f_ks = np.array(E[contuis_feature])
        f_ks.sort()
        k = len(f_ks)
        new_features = range(0,k-1)
        for f in new_features:
            sum = self.IG_binary((f_ks[f] + f_ks[f+1])/2 , contuis_feature,feature_index,E)
            if sum > best_sum:
                best_sum = sum
                best_saf = (f_ks[f] + f_ks[f+1])/2

        return  best_sum , best_saf



    def IG_max(self,E,F):
        assert isinstance(E,pandas.DataFrame)
        max = -float('inf')
        feature = None
        saf = 0
        f_index = 1
        for i in F:
            IG_val , best_saf = self.IG(i,f_index,E)
            f_index+=1
            if IG_val >= max:
                max = IG_val
                feature = i
                saf = best_saf
        return feature,saf

    def H(self,E):
        sum = 0
        eps = np.finfo(float).eps
        assert isinstance(E,pandas.DataFrame)
        B,M = split_data_frame(E,None,None,True)
        for c in ['B','M']:
            size_c  = len(B if c == 'B' else M)
            size_table = len(E)

            prob_c = size_c/size_table
            log =  math.log(prob_c + eps,2)
            sum+= -prob_c * log
        return sum
