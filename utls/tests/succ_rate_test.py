

def test(df_test,Classify_func):
    correct = 0
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
        if predicted_c == actual_c:
            correct+= 1
    return correct / len(test_data)