

def loss_rate(df_test,Classify_func):
    FP=0
    FN=0
    test_data = df_test
    features = test_data.columns[1::].values.tolist()
    index = 1
    features_dict = {}
    for f in features:
        features_dict[f] = index
        index += 1

    for i in range(0,len(test_data)):
        test_sample = test_data.iloc[i].tolist()
        actual_c = test_sample[0]
        predicted_c = Classify_func(test_sample , features_dict)

        if actual_c == 'B' and predicted_c =='M':
            FP += 1
        if actual_c == 'M' and predicted_c == 'B':
            FN += 1


    return (0.1 * FP + FN) / len(test_data)



