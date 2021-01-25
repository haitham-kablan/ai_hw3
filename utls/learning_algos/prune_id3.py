import utls.TDIDT as TDIDT
import utls.np_array_helper_functions as np_utls

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