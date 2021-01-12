import utls.TDIDT as UT
import utls.SelectFeatures
import pandas

class ID3:
    def __init__(self , file_name):
        data = pandas.read_csv(file_name)
        self.features = data.columns[1::].values.tolist()
        Default = data.diagnosis.mode()[0]
        self.tree = utls.TDIDT.TDIT(data,self.features,Default,utls.SelectFeatures.IG_MAX().IG_max)

    def Classify(self , o , features_dict):
        return utls.TDIDT.Classify(self.tree,o , features_dict)




