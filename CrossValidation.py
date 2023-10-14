from functools import reduce as rd
import pandas as pd
import os
from itertools import product
from functools import partial as pf
import time
from copy import copy
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
        pass

    def getErrorDf(self, train_dict, bin_number, p_val, m_val):
        def error(i):
            (b, p, m) = my_space[i]
            binned_df = self.nb.binned(b)
            Qframe = self.nb.getQ(binned_df)
            Fframe = self.nb.getFs(m, p, binned_df)
            pred_class = self.nb.predicted_class(Qframe, Fframe)
            value = self.nb.value(binned_df)
            predicted_classes = binned_df.index.map(lambda i: pred_class(value(i)))
            actual_classes = binned_df.index.map(self.nb.target)

        start_time = time.time()
        folds = pd.Index(range(10))
        my_space = pd.Series(product(bin_number, p_val, m_val))
        output_col_titles = ["Zero_One_Loss_Avg"]
        df_size = len(my_space)
        cols = list(zip(*my_space))
        col_titles = ["Bin Number", "p_val", "m_val"]
        data = zip(col_titles, cols)
        error_df = pd.DataFrame(index = range(df_size))
        for (title, col) in data:
            error_df[title] = col
        for title in output_col_titles:
            error_df[title] = df_size * [None]
        tuples = pd.Series(range(df_size)).map(error).values
        output_cols = list(zip(*tuples))
        output_data = zip(output_col_titles, output_cols)
        for (title, col) in output_data:
            error_df[title] = col
        error_df.to_csv(os.getcwd() + '\\' + str(self.data) + '\\' + "{}_Error.csv".format(str(self.data)))
        print("Time Elapsed: {} Seconds".format(time.time() - start_time))
        return error_df



    def test(self):
        p = self.partition(10)
        (train_dict, test_dict) = self.training_test_dicts(self.data.df, p)
        #error_df = self.getErrofDf

