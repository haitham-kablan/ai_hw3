

import pandas
from sklearn.model_selection import KFold
import numpy as np
import utls.learning_algos.ID3_impl as ID3_impl
import utls.tests.succ_rate_test as succ_rate_test
def get_local_best_feature(F):

    data = pandas.read_csv('train.csv')
    kf = KFold(n_splits=5, shuffle=True, random_state=209418441)
    avg_per_feature = np.zeros(len(F))
    for train_index, test_index in kf.split(data):
        f_index = 0
        for f in F:
            train = (data.iloc[train_index])[['diagnosis' , f]]

            test = data.iloc[test_index]

            id3_tree = ID3_impl.ID3(train,0)
            success_rate = succ_rate_test.test(test,id3_tree.Classify)
            avg_per_feature[f_index] += success_rate
            print('feature: ',f,' accuray is: ',success_rate)
        print("     ###################")
    return np.argmax(avg_per_feature) + 1











if __name__ == '__main__':
    df = pandas.read_csv('train.csv')
    features = df.columns[1::].values.tolist()
    print(features[get_local_best_feature(features)])