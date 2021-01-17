import numpy as np

def test(df_test,Classify_func):

    correct = 0
    test_data = np.array(df_test)

    for i in range(0,len(test_data)):

        test_sample = test_data[i]
        actual_c = test_sample[0]
        predicted_c = Classify_func(test_sample)
        if predicted_c == actual_c:
            correct+= 1
    return correct / len(test_data)