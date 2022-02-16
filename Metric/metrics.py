import numpy as np
import pandas as pd

# TODO: Multi-Label update

def accuracy_score(y_test, pred):
    return np.mean(np.argmax(pred, axis=1) == y_test)

def recall_score(y_test, pred, multi_label=False):
    tp, _, _, fn = calc_labels(y_test, pred, multi_label)

    return tp / (tp + fn)

def precision_score(y_test, pred, multi_label=False):
    tp, fp, _, _ = calc_labels(y_test, pred, multi_label)

    return tp / (tp +fp)


def f1_score(y_test, pred, multi_label=False):
    pr = precision_score(y_test, pred, multi_label)
    rc = recall_score(y_test, pred, multi_label)

    return 2 * (pr * rc) / (pr + rc)


def confusion_matrix(y_test, pred, multi_label=False):
    tp, fp, tn, fn = calc_labels(y_test, pred, multi_label)
    
    return np.array([tp, fp],[fn, tn])


def calc_labels(y_test, pred, multi_label):
    if not multi_label: 
        tn = 0
        fn = 0
        tp = 0
        fp = 0
        
        for gt, ps in zip(y_test, pred):
            if gt and ps:
                tp += 1
            elif gt and not ps:
                fn += 1
            elif not gt and not ps:
                tn += 1
            elif not gt and ps:
                fp += 1

        return tp, fp, tn, fn