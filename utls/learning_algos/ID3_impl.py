import utls.TDIDT as UT
import utls.SelectFeatures
import pandas
import numpy as np
import utls.np_array_helper_functions as np_utls

class ID3:
    def __init__(self, df,M):
        assert isinstance(df,pandas.DataFrame)
        self.df_as_np_array = np.array(df)
        self.features = df.columns[1::].values.tolist()

        Default = np_utls.calc_majority(self.df_as_np_array)

        self.tree = utls.TDIDT.TDIT(self.df_as_np_array,self.features,Default,utls.SelectFeatures.IG_MAX().IG_max,M)

    def Classify(self , o):
        return utls.TDIDT.Classify(self.tree,o)



