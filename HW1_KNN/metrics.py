import numpy as np


def binary_classification_metrics(y_pred, y_true):
    """
    Computes metrics for binary and multiclass classification
    Arguments:
    y_pred, np array (num_samples) - model predictions
    y_true, np array (num_samples) - true labels
    Returns:
    precision, recall, f1, accuracy - classification metrics
    """

    # TODO: implement metrics!
    # Some helpful links:
    # https://en.wikipedia.org/wiki/Precision_and_recall
    # https://en.wikipedia.org/wiki/F1_score
    
    accuracy = sum([1 for y_p, y_t in zip(y_pred, y_true) if y_p == y_t]) / len(y_pred)
    
    tp = sum([1 for y_p, y_t in zip(y_pred, y_true) if y_p == "1" and y_t == "1"])
    fp = sum([1 for y_p, y_t in zip(y_pred, y_true) if y_p == "1" and y_t == "0"])
    tn = sum([1 for y_p, y_t in zip(y_pred, y_true) if y_p == "0" and y_t == "0"])
    fn = sum([1 for y_p, y_t in zip(y_pred, y_true) if y_p == "0" and y_t == "1"])
    
    precision = tp / (fp + tp) if (fp + tp) != 0 else 0
     
    recall = tp / (fn + tp) if (fn + tp) != 0 else 0
    
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) != 0 else 0
    
    return accuracy, precision, recall, f1


def multiclass_accuracy(predictions, y_test):
    
    return sum([1 for y_p, y_t in zip(predictions, y_test) if y_p == y_t]) / len(predictions)


def r_squared(y_pred, y_true):
    """
    Computes r-squared for regression
    Arguments:
    y_pred, np array of int (num_samples) - model predictions
    y_true, np array of int (num_samples) - true values
    Returns:
    r2 - r-squared value
    """

    y_mean = y_true.mean()
    tss = ((y_pred - y_mean) ** 2).sum()
    rss = ((y_pred - y_true) ** 2).sum()
    return 1 - rss / tss
    

def mse(y_pred, y_true):
    """
    Computes mean squared error
    Arguments:
    y_pred, np array of int (num_samples) - model predictions
    y_true, np array of int (num_samples) - true values
    Returns:
    mse - mean squared error
    """    
    return np.mean((y_pred - y_true) ** 2)


def mae(y_pred, y_true):
    """
    Computes mean absolut error
    Arguments:
    y_pred, np array of int (num_samples) - model predictions
    y_true, np array of int (num_samples) - true values
    Returns:
    mae - mean absolut error
    """
    return np.mean(np.abs(y_true - y_pred))
    