import pandas
import math
import numpy
import utls


class IG_MAX:

    def IG_binary(self,saf,continous_feature,E):
        assert isinstance(E,pandas.DataFrame)
        H_e = self.H(E)
        E_size = len(E.index)
        sum = 0
        for c in [0,1]:
            e_i = None
            if c == 0:
                e_i = E.loc[E[continous_feature] >= saf ]
            else:
                e_i = E.loc[E[continous_feature] < saf]
            e_i_size = len(e_i.index)
            if e_i_size == 0:
                return float('-inf')
            h_e_i = self.H(e_i)
            sum += (e_i_size/E_size) * h_e_i
        return H_e - sum




    def IG(self,contuis_feature,E):

        assert isinstance(E,pandas.DataFrame)
        h_e = self.H(E)
        best_sum = -float('inf')
        best_saf = -float('inf')
        f_ks = numpy.array(E[contuis_feature])
        f_ks.sort()
        k = len(f_ks)
        new_features = range(0,k-1)
        for f in new_features:
            sum = self.IG_binary((f_ks[f] + f_ks[f+1])/2 , contuis_feature,E)
            if sum > best_sum:
                best_sum = sum
                best_saf = (f_ks[f] + f_ks[f+1])/2

        return  best_sum , best_saf



    def IG_max(self,E,F):
        assert isinstance(E,pandas.DataFrame)
        max = -float('inf')
        feature = None
        saf = 0
        for i in F:
            IG_val , best_saf = self.IG(i,E)
            if IG_val >= max:
                max = IG_val
                feature = i
                saf = best_saf
        return feature,saf

    def H(self,E):
        sum = 0
        eps = numpy.finfo(float).eps
        assert isinstance(E,pandas.DataFrame)
        for c in ['B','M']:
            size_c  = len(E.loc[E['diagnosis'] == c].index)
            size_table = len(E.index)

            prob_c = size_c/size_table
            log =  math.log(prob_c + eps,2)
            sum+= -prob_c * log
        return sum
