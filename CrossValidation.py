from functools import reduce
import pandas as pd
import os
from itertools import product
import time
from NaiveBayes import NaiveBayes

class CrossValidation:
    def __init__(self, data):
        self.data = data
        self.nb = NaiveBayes(data)

    def partition(self, k):
        n = self.data.df.shape[0]
        (q, r) = (n // k, n % k)
        def f(i, j, p):
            return p if i == k else f(i + 1, j + q + int(i < r), p + [list(range(j, j + q + int(i < r)))])
        return f(0, 0, [])

    def training_test_dicts(self, df, partition=None):
        partition = self.partition(10) if partition is None else partition
        train_index = lambda i: reduce(lambda l1, l2: l1 + l2, partition[:i] + partition[i + 1:])
        test_dict = dict([(i, df.filter(items=partition[i], axis=0)) for i in range(len(partition))])
        train_dict = dict([(i, df.filter(items=train_index(i), axis=0)) for i in range(len(partition))])
        return (train_dict, test_dict)

    def zero_one_loss(self, predicted, actual):
        return sum(predicted.to_numpy() != actual.to_numpy()) / len(predicted)

    def getErrorDf(self, train_dict, bin_numbers, p_vals, m_vals):
        def error(f, b, p, m):
            binned_df = self.nb.binned(train_dict[f], b)
            pred_class = self.nb.predicted_class(binned_df, p, m)
            predicted_classes = binned_df.index.map(lambda i: pred_class(self.nb.value(binned_df)(i)))
            actual_classes = binned_df.index.map(self.nb.target)
            return self.zero_one_loss(predicted_classes, actual_classes)

        start_time = time.time()
        folds = pd.Index(range(10))
        my_space = pd.Series(product(folds, bin_numbers, p_vals, m_vals)).map(lambda hyps: hyps + (error(*hyps),))
        col_titles = ["Fold", "Bin Number", "p_val", "m_val", "Error"]
        error_df = pd.DataFrame.from_dict(data = dict(my_space), orient = "index", columns = col_titles)
        error_df.to_csv("\\".join([os.getcwd(), str(self.data), "{}_Error.csv".format(str(self.data))]))
        print("Time Elapsed: {} Seconds".format(time.time() - start_time))
        return error_df


    def test(self, bin_numbers, p_vals, m_vals):
        p = self.partition(10)
        (train_dict, test_dict) = self.training_test_dicts(self.data.df, p)
        error_df = self.getErrorDf(train_dict, bin_numbers, p_vals, m_vals)

