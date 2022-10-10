import numpy as np
from sklearn.metrics import accuracy_score, precision_score

def perc_diff_from_best_global(outputs, y_test, y_range_test):
    diff = []
    for output, label_score, score_range in zip(outputs, y_test, y_range_test):
        best, worst = score_range
        m = abs(best - label_score[output]) / (best - worst)
        diff.append(m)
    return np.nanmean(diff)

def perc_diff_from_worst_global(outputs, y_test, y_range_test):
    diff = []
    for output, label_score, score_range in zip(outputs, y_test, y_range_test):
        best, worst = score_range
        m = abs(label_score[output] - worst) / (best - worst)
        diff.append(m)
    return np.nanmean(diff)

def perc_diff_from_best_subset(labels, outputs, y_test, preds):
    diff = []
    for label, output, label_score, output_score in zip(labels, outputs, y_test, preds):
        m = abs(label_score[label] - label_score[output]) / (label_score[label] - min(label_score))
        diff.append(m)
    return np.nanmean(diff)

def perc_diff_from_worst_subset(labels, outputs, y_test, preds):
    diff = []
    for label, output, label_score, output_score in zip(labels, outputs, y_test, preds):
        m = abs(label_score[output] - min(label_score)) / (label_score[label] - min(label_score))
        diff.append(m)
    return np.nanmean(diff)

def get_mae(labels, outputs, y_test, preds):
    mae = []
    for label, output, label_score, output_score in zip(labels, outputs, y_test, preds):
        mae.append(np.mean(np.abs(label_score - output_score)))
    return np.nanmean(mae)

def get_perf_of_best_predicted(labels, outputs, y_test, preds):
    perf_best_predicted = []
    for label, output, label_score, output_score in zip(labels, outputs, y_test, preds):
        perf_best_predicted.append(label_score[output])
    return np.nanmean(perf_best_predicted)

def get_metrics(y_test, y_best_test, preds):
    metrics = {}
    labels = [np.argmax(yt) for yt in y_test]
    outputs = [np.argmax(p) for p in preds]

    # metrics['precision'] = np.mean(precision_score(labels, outputs, average=None))
    metrics['accuracy'] = accuracy_score(labels, outputs)
    metrics["perc_diff_from_best_global"] = perc_diff_from_best_global(outputs, y_test, y_best_test)
    #metrics["perc_diff_from_worst_global"] = perc_diff_from_worst_global(outputs, y_test, y_range_test)
    metrics['perc_diff_from_best_subset'] = perc_diff_from_best_subset(labels, outputs, y_test, preds)
    #metrics['perc_diff_from_worst_subset'] = perc_diff_from_worst_subset(labels, outputs, y_test, preds)
    metrics['mae'] = get_mae(labels, outputs, y_test, preds)
    metrics['perf_of_best_predicted'] = get_perf_of_best_predicted(labels, outputs, y_test, preds)
    return metrics

