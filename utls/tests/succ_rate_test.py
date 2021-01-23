import numpy as np

def test(df_test,Classify_func):
    """
    this function will calculate the success rate for a given classifier on the given test
    :param df_test: the test it self
    :param Classify_func: the classify function of the classifier that we want to test.
    :return: the success rate of the given classifier on the test
    """

    correct = 0
    test_data = np.array(df_test)

    for i in range(0,len(test_data)):

        test_sample = test_data[i]
        actual_c = test_sample[0]
        predicted_c = Classify_func(test_sample)
        if predicted_c == actual_c:
            correct+= 1
    return correct / len(test_data)