
import pandas


def all_same(column , item):
    a = column.to_numpy()
    return (item == a).all()


class ID3:
    def __init__(self,SelectFeature , features):
        self.SelectFeature = SelectFeature
        self.features = features

    def TDIDT(self,E,F,Default):

        assert isinstance(E,pandas.DataFrame)

        if E.empty:
            return None, {}, Default

        c = E.diagnosis.value_counts().idxmax()
        all_c = all_same(E['diagnosis'],c)

        if all_c or F=={}:
            return None,{},c

        new_feature = self.SelectFeature(F,E)
        F = self.features - new_feature

        subtrees = None
        for value in new_feature.domain:
            self.TDIDT(E/f(v),F,c)


