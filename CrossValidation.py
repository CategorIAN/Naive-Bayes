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
        def error(i):
            (f, b, p, m) = my_space[i]
            binned_df = self.nb.binned(train_dict[f], b)
            Qframe = self.nb.getQ(binned_df)
            Fframe = self.nb.getFs(m, p, binned_df)
            pred_class = self.nb.predicted_class(Qframe, Fframe)
            predicted_classes = binned_df.index.map(lambda i: pred_class(self.nb.value(binned_df)(i)))
            actual_classes = binned_df.index.map(self.nb.target)
            return self.zero_one_loss(predicted_classes, actual_classes)

        start_time = time.time()
        folds = pd.Index(range(10))
        my_space = pd.Series(product(folds, bin_numbers, p_vals, m_vals))
        print(my_space)
        output_col_titles = ["Error"]
        df_size = len(my_space)
        cols = list(zip(*my_space))
        col_titles = ["Fold", "Bin Number", "p_val", "m_val"]
        data = zip(col_titles, cols)
        error_df = pd.DataFrame.from_dict(data = dict(my_space), orient = "index", columns = col_titles)

        for title in output_col_titles:
            error_df[title] = df_size * [None]
        tuples = pd.Series(range(df_size)).map(error).values
        output_cols = [tuples]
        output_data = zip(output_col_titles, output_cols)
        for (title, col) in output_data:
            error_df[title] = col
        """
        error_df.to_csv(os.getcwd() + '\\' + str(self.data) + '\\' + "{}_Hyps.csv".format(str(self.data)))
        print("Time Elapsed: {} Seconds".format(time.time() - start_time))
        return error_df


    def test(self, bin_numbers, p_vals, m_vals):
        p = self.partition(10)
        (train_dict, test_dict) = self.training_test_dicts(self.data.df, p)
        error_df = self.getErrorDf(train_dict, bin_numbers, p_vals, m_vals)

