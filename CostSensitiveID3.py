import utls.np_array_helper_functions as np_utls
import utls.TDIDT as TDIDT
import utls.learning_algos.ID3_impl as ID3_impl
import pandas as pd
import utls.tests.loss_rate_test as loss_rate_test
from sklearn.model_selection import KFold
import numpy as np
import math


class prune_ID3:
    def __init__(self, Tree, V):
        self.prune_tree = prune(Tree, V)

    def Classify(self, o):
        return TDIDT.Classify(self.prune_tree, o)


def prune(Tree, V):
    feature_index, saf, subtrees, c = Tree

    if len(subtrees) == 0:
        return Tree

    samples = np_utls.split(V, feature_index, saf)

    for i in range(0, 2):
        subtrees[i] = prune(subtrees[i], samples[i])

    err_prune = 0
    err_no_prune = 0

    # calc eorrs
    for row in V:
        actual_c = row[0]
        predicted_c = TDIDT.Classify(Tree, row)
        err_prune += Evaluate(actual_c, c)
        err_no_prune += Evaluate(actual_c, predicted_c)

    if err_prune < err_no_prune:
        Tree = None, None, [], c

    return Tree


def Evaluate(actual_c, predicted_c):
    if actual_c == predicted_c:
        return 0
    else:
        return 1 if (actual_c == 'B' and predicted_c == 'M') else 10


def prune_expr(data, ratio):
    assert isinstance(data, pd.DataFrame)
    VAL_PROB = [0.1, 0.2, 0.25, 0.3, 0.35, 0.4]
    n = len(data)
    avg = []
    for val_prop in VAL_PROB:
        for p in ratio:
            avg.append([0 , (p,val_prop)])
    index = 1
    kf = KFold(n_splits=5, shuffle=True, random_state=209418441)
    for train_index, test_index in kf.split(data):
        print('testing index = ', index)
        p_index = 0
        for val_prob in VAL_PROB:

            for p in ratio:
                train = data.iloc[train_index]
                test = data.iloc[test_index]
                validation_data = train.sample(math.floor(len(train) * val_prob))
                train.drop(validation_data.index.tolist())
                data = pd.read_csv('train.csv')
                id3_before_prune = ID3_impl.ID3(train, 0, p)
                id3_after_prune = prune_ID3(id3_before_prune.tree, validation_data.values)
                loss_rate = loss_rate_test.loss_rate(test, id3_after_prune.Classify)
                print('     loss rate for p= ', p, ' val_prob= ', val_prob, 'loss_rate is: ', loss_rate)
                avg[p_index][0] += loss_rate
                p_index += 1
        break
    return min(avg, key=lambda x: x[0])


def experiment(data, RATIO_VALS):
    assert isinstance(data, pd.DataFrame)
    kf = KFold(n_splits=5, shuffle=True, random_state=209418441)
    avg_per_ratio = np.zeros(len(RATIO_VALS))
    out_index = 1
    for train_index, test_index in kf.split(data):
        print('         Testing for index: ', out_index)
        train = data.iloc[train_index]
        test = data.iloc[test_index]
        index = 0
        out_index += 1
        for ratio in RATIO_VALS:
            id3_tree = ID3_impl.ID3(train, 0, ratio)
            loss_rate = loss_rate_test.loss_rate(test, id3_tree.Classify)
            avg_per_ratio[index] += loss_rate
            index += 1
            print('             for ratios= ', ratio, ' current avg loss rate is: ', loss_rate / out_index)
    return min(avg_per_ratio)


if __name__ == '__main__':

    df_data = pd.read_csv('train.csv')
    test = pd.read_csv('test.csv')
    RATIO_VALS = []
    for i in range(0, 10):
        RATIO_VALS.append(i / 10)
    print(RATIO_VALS)
    normal_loss_rate = loss_rate_test.loss_rate(test, ID3_impl.ID3(df_data, 0, 1.5).Classify)
    n = len(df_data)
    best = prune_expr(df_data, RATIO_VALS)

    print('     DONE EXPERIMENT **** best ratio is for p = ', best[1][0], ' with val: ', best[1][1],
          ' and loss rate = ', best[0])
    print('nrml loss is: ', normal_loss_rate)

    validation_data = df_data.sample(math.floor(len(df_data) * best[1][1]))
    train_data = df_data.drop(validation_data.index.tolist())
    id3_before_prune = ID3_impl.ID3(train_data, 0, best[1][0])
    id3_after_prune = prune_ID3(id3_before_prune.tree, validation_data.values)
    loss_rate = loss_rate_test.loss_rate(test, id3_after_prune.Classify)
    print('new loss rate is: ', loss_rate)
    print('shebor is: ', normal_loss_rate / loss_rate)
