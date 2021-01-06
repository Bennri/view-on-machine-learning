import numpy as np
from sklearn.metrics import f1_score


def f1_score_helper(y_true, y_preds, c_label):
    idx = y_true == c_label
    curr_y_true = y_true[idx]
    curr_y_preds = y_preds[idx]
    tp = np.sum(curr_y_true == curr_y_preds)
    fp = np.sum(y_preds[y_true != c_label] == c_label)
    fn = np.sum(curr_y_true != curr_y_preds)
    res = np.divide(2 * tp, 2 * tp + fp + fn)
    return res


def f1_score_impl(y_true, y_preds, average=None, pos_label=1):
    n_classes = np.unique(y_true)
    results = np.zeros(n_classes.shape)
    for i, c in enumerate(n_classes):
        results[i] = f1_score_helper(y_true, y_preds, c)
    # two class task and only one score is requested
    if len(n_classes) < 3 and average is 'binary':
        return results[np.where(n_classes == pos_label)[0]][0]
    else:
        return results


if __name__ == '__main__':
    Y_true = np.array([0, 0, 1, 1, 1, 0])
    Y_preds = np.array([0, 1, 1, 1, 0, 1])

    print(f1_score(Y_true, Y_preds, average=None))
    print(f1_score_impl(Y_true, Y_preds, average=None))
    # see: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html
    # default value of average: binary
    # only the results for the positive label (default=1) is returned
    print(f1_score(Y_true, Y_preds))
    print(f1_score_impl(Y_true, Y_preds, average='binary'))

    # multiclass
    Y_true = np.array([0, 0, 1, 1, 1, 0, 2, 2])
    Y_preds = np.array([0, 1, 1, 1, 0, 1, 2, 0])

    print(f1_score(Y_true, Y_preds, average=None))
    # here, the value of average is not taken into account due to multiclass problem
    print(f1_score_impl(Y_true, Y_preds, average='binary'))
