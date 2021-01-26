import utls.TDIDT as UT
import utls.SelectFeatures
import pandas
import numpy as np
import utls.np_array_helper_functions as np_utls


class ID3:
    def __init__(self, df,M,improved = False,ratio = 1.5):
        """
        this class will bulid for us an id3 decision tree and store it in self.tree
        :param df: the train data
        :param M: the prune parameter
        :param ratio: this is used for part 4 (improving the loss) if it bigger than 1
        then this class will behave exactly as the id3 that we learned in the lectures
        (it is not relevant for part 1 and 3)
        """
        assert isinstance(df,pandas.DataFrame)
        self.df_as_np_array = np.array(df)
        self.features = df.columns[1::].values.tolist()

        Default = np_utls.calc_majority(self.df_as_np_array)

        self.tree = utls.TDIDT.TDIDT(self.df_as_np_array,self.features,Default,utls.SelectFeatures.IG_MAX().IG_max,M,improved=improved,ratio=ratio)

    def Classify(self , o):
        return utls.TDIDT.Classify(self.tree,o)





