from functools import reduce
import pandas as pd
import os
from itertools import product
import time
from NaiveBayes import NaiveBayes
from dataclasses import dataclass
from MLData import Dataset
from Error import zero_one_loss

@dataclass(frozen=True)
class CrossValidation:
    data: Dataset
    hyp_range: dict[str, [str]]

    def partition(self) -> [[int]]:
        n = self.data.X.shape[0]
        (q, r) = (n // 10, n % 10)

        def f(i, j, p):
            if i == 10:
                return p
            else:
                return f(i + 1, j + q + int(i < r), p + [list(range(j, j + q + int(i < r)))])
        return f(0, 0, [])

    def training_test_dict(self) -> dict[int, tuple[Dataset, Dataset]]:
        p = self.partition()
        train = lambda i: self.data.filter(reduce(lambda l1, l2: l1 + l2, p[:i] + p[i + 1:]))
        test = lambda i: self.data.filter(p[i])
        return {i: (train(i), test(i)) for i in range(10)}


    def error(self, train_dict, test_dict):
        def error_func(f, b, p, m):
            train_df, test_df = (self.nb.binned(train_dict[f], b), self.nb.binned(test_dict[f], b))
            pred_class = self.nb.predicted_class(train_df, p, m)
            predicted_classes = test_df.index.map(lambda i: pred_class(self.nb.value(test_df)(i)))
            actual_classes = test_df.index.map(self.nb.target)
            return self.zero_one_loss(predicted_classes, actual_classes)
        return error_func

    def error(self, model: NaiveBayes, train: Dataset, test:Dataset):
        classifier = model(train)
        predicted = test.X.apply(classifier, axis=1)
        actual = test.y
        return zero_one_loss(predicted, actual)

    def getErrorDf(self, train_dict, test_dict, bin_numbers, p_vals, m_vals):
        start_time = time.time()
        folds = pd.Index(range(10))
        error_func = self.error(train_dict, test_dict)
        rows = pd.Series(product(folds, bin_numbers, p_vals, m_vals)).map(lambda hyps: hyps + (error_func(*hyps),))
        col_titles = ["Fold", "Bin Number", "p_val", "m_val", "Error"]
        error_df = pd.DataFrame.from_dict(data = dict(rows), orient = "index", columns = col_titles)
        error_df.to_csv("\\".join([os.getcwd(), str(self.data), "{}_Error.csv".format(str(self.data))]))
        print("Time Elapsed: {} Seconds".format(time.time() - start_time))
        return error_df

    def getAnalysisDf(self, error_df):
        analysis_df = error_df.groupby(by = ["Bin Number", "p_val", "m_val"]).mean()[["Error"]]
        analysis_df.to_csv("\\".join([os.getcwd(), str(self.data), "{}_Analysis.csv".format(str(self.data))]))
        return analysis_df

    def best_params(self, bin_numbers, p_vals, m_vals):
        p = self.partition(10)
        (train_dict, test_dict) = self.training_test_dicts(self.data.df, p)
        error_df = self.getErrorDf(train_dict, test_dict, bin_numbers, p_vals, m_vals)
        analysis_df = self.getAnalysisDf(error_df)
        best_row = analysis_df.loc[lambda df: df["Error"] == analysis_df["Error"].min()].iloc[0]
        return best_row

    def value(self, df):
        return lambda i: df.loc[i, self.data.features]

    def target(self, i):
        return self.data.df.at[i, "Class"]



