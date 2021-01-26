
def minimize_loss(E,F):

    max = -float('inf')
    saf = 0
    best_feature = 0
    # starts from 1 so we dont take diagnosis
    f_index = 1
    for i in F:
        IG_val, best_saf = self.IG(f_index, E)

        if IG_val >= max:
            max = IG_val
            best_feature = f_index
            saf = best_saf
        f_index += 1
    return best_feature, saf