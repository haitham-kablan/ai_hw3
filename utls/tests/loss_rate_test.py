

def loss_rate(df_test,Classify_func):

    """
    this function will calculate the loss rate of the given classifier for the given test
    :param df_test: the test it self
    :param Classify_func: the classify function of the classifier that we want to test
    :return: the loss rate of the given classifier on the test
    """
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
        predicted_c = Classify_func(test_sample)

        if actual_c == 'B' and predicted_c =='M':
            FP += 1
        if actual_c == 'M' and predicted_c == 'B':
            FN += 1


    return (0.1 * FP + FN) / len(test_data)



