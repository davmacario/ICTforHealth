import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.cluster as sk

def evalAccuracy(col_predict, col_label):
    if (len(col_predict.shape) > 1 or len(col_label.shape) > 1):
        if (not all(col_predict.shape[1:] == 1) or not all(col_label.shape[1:] == 1)):
            raise ValueError("The passed arrays are not 1D")
    
    n_elem = col_predict.shape[0]
    if (col_label.shape[0] != n_elem):
        raise ValueError("The two arrays don't have the same length!")

    tmp_sum = 0
    for i in range(n_elem):
        if col_label[i] == col_predict[i]:
            tmp_sum += 1

    accuracy = tmp_sum/n_elem

    return accuracy
