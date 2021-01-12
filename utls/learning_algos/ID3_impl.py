import utls.TDIDT as UT
import utls.SelectFeatures
import pandas

class ID3:
    def __init__(self , file_name):
        data = pandas.read_csv(file_name)
        self.features = data.columns[1::].values.tolist()
        Default = data.diagnosis.mode()[0]
        self.tree = utls.TDIDT.TDIT(data,self.features,Default,utls.SelectFeatures.IG_MAX().IG_max)

    def Classify(self , o):
        return utls.TDIDT.Classify(self.tree,o)


    def test(self,test_file_name):
        correct = 0
        test_data = pandas.read_csv(test_file_name)
        features = test_data.columns[1::].values.tolist()
        for i in range(0,len(test_data)):
            test_sample = {}
            for feature in self.features:
                test_sample[feature] = test_data[feature][i]
            c = self.Classify(test_sample)
            if c == test_data['diagnosis'][i]:
                correct+= 1
        print(correct / len(test_data))

