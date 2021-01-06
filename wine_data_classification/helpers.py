import numpy as np
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score


def compute_metric_components(y_true, y_preds, c_label):
    idx = y_true == c_label
    curr_y_true = y_true[idx]
    curr_y_preds = y_preds[idx]
    tp = np.sum(curr_y_true == curr_y_preds)
    fp = np.sum(y_preds[y_true != c_label] == c_label)
    fn = np.sum(curr_y_true != curr_y_preds)
    tn = np.sum(curr_y_true != curr_y_preds)
    return tp, fp, tn, fn


def accuracy(y_true, y_preds):
    z = y_true == y_preds
    return np.count_nonzero(z) / len(y_true)


def precision_helper(y_true, y_preds, c_label):
    tp, fp, tn, fn = compute_metric_components(y_true, y_preds, c_label)
    return np.divide(tp, tp + fp)


def precision(y_true, y_preds):
    n_classes = np.unique(y_true)
    results = np.zeros(n_classes.shape)
    for i, c in enumerate(n_classes):
        results[i] = precision_helper(y_true, y_preds, c)
    return results


def recall_helper(y_true, y_preds, c_label):
    tp, fp, tn, fn = compute_metric_components(y_true, y_preds, c_label)
    return np.divide(tp, tp + fn)


def recall(y_true, y_preds):
    n_classes = np.unique(y_true)
    results = np.zeros(n_classes.shape)
    for i, c in enumerate(n_classes):
        results[i] = recall_helper(y_true, y_preds, c)
    return results


def f1_score_helper(y_true, y_preds, c_label):
    tp, fp, _, fn = compute_metric_components(y_true, y_preds, c_label)
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
    print('F1-Score')
    print('#' * 80)
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
    print()

    print('Accuracy')
    print('#' * 80)
    Y_true = np.array([0, 0, 1, 1, 1, 0])
    Y_preds = np.array([0, 1, 1, 1, 0, 1])
    print(accuracy_score(Y_true, Y_preds))
    print(accuracy(Y_true, Y_preds))

    # multiclass
    Y_true = np.array([0, 0, 1, 1, 1, 0, 2, 2])
    Y_preds = np.array([0, 1, 1, 1, 0, 1, 2, 0])

    print(accuracy_score(Y_true, Y_preds))
    print(accuracy(Y_true, Y_preds))

    # multiclass
    Y_true = np.array([0, 0, 1, 1, 1, 1, 1, 0, 2, 2, 2, 2, 2, 0, 1])
    Y_preds = np.array([0, 1, 1, 1, 1, 1, 0, 1, 2, 0, 2, 2, 2, 0, 1])

    print(accuracy_score(Y_true, Y_preds))
    print(accuracy(Y_true, Y_preds))

    print('Recall')
    print('#' * 80)
    print(recall_score(Y_true, Y_preds, average=None))
    print(recall(Y_true, Y_preds))
    print()

    print('Precision')
    print('#' * 80)
    print(precision_score(Y_true, Y_preds, average=None))
    print(precision(Y_true, Y_preds))
