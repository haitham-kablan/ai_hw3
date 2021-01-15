
import pandas
from sklearn.model_selection import KFold
import utls.learning_algos.ID3_impl as ID3_imp
import utls.tests.succ_rate_test as run_test

def aplly_k_validation(file_name):
    M_list= [3,13,33,73,123]
    succ_rate = [0,0,0,0,0]
    data = pandas.read_csv(file_name)
    kf = KFold(n_splits=5, shuffle=True, random_state=209418441)
    for i in range(0,5):
        result = next(kf.split(data,None))
        train = data.iloc[result[0]]
        df_test = data.iloc[result[1]]
        index = 0
        for M in M_list:
            print('testing M :',M)
            Classifer_ID3 = ID3_imp.ID3(train,M)
            success_rate = run_test.test(df_test, Classifer_ID3.Classify)
            print('i is',i,'M is',M , 'succ_rate' , success_rate)
            succ_rate[index] += success_rate
            index +=1
    avg_succ_rate = [i/5 for i in succ_rate]
    print(avg_succ_rate)
    return avg_succ_rate

