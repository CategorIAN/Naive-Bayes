import pandas as pd
import numpy as np
from copy import copy, deepcopy
import random
import math
from functools import reduce

class NaiveBayes:
    def __init__(self, data):
        self.data = data
        self.seed = random.random()

    def binned(self, n):
        def f(col):
            try:
                colvalues = self.data.df[col].apply(pd.to_numeric)
                return pd.qcut(colvalues.rank(method="first"), q=n, labels=range(n))
            except:
                return self.data.df[col]
        return pd.DataFrame(dict([(col, f(col)) for col in self.data.df.columns]))

    def noised(self):
        random.seed(self.seed)
        noise_features = random.sample(self.data.features, k=math.ceil(len(self.data.features) * .1))
        def f(col):
            return pd.Series(np.random.permutation(self.data.df[col])) if col in noise_features else self.data.df[col]
        return pd.DataFrame(dict([(col, f(col)) for col in self.data.df.columns]))

    def partition(self, k):
        n = self.data.df.shape[0]
        (q, r) = (n // k, n % k)
        def f(i, j, p):
            return p if i == k else f(i + 1, j + q + int(i < r), p + [list(range(j, j + q + int(i < r)))])
        return f(0, 0, [])

    def training_test_sets(self, j, df, partition=None):
        if partition is None: partition = self.partition(10)
        train = []
        for i in range(len(partition)):
            if j != i:
                train += partition[i]
            else:
                test = partition[i]
        self.train_set = df.filter(items=train, axis=0)
        self.test_set = df.filter(items=test, axis=0)

    def training_test_dicts(self, df, partition=None):
        partition = self.partition(10) if partition is None else partition
        train_index = lambda i: reduce(lambda l1, l2: l1 + l2, partition[:i] + partition[i + 1:])
        test_dict = dict([(i, df.filter(items=partition[i], axis=0)) for i in range(len(partition))])
        train_dict = dict([(i, df.filter(items=train_index(i), axis=0)) for i in range(len(partition))])
        return (train_dict, test_dict)

    def getQ(self, train_dict = None):
        train_dict = self.training_test_dicts(self.data.df)[0] if train_dict is None else train_dict
        def f(i):
            train = train_dict[i]
            df = pd.DataFrame(train.groupby(by = ["Class"])["Class"].agg("count")).rename(columns={"Class": "Count"})
            return pd.concat([df, pd.Series(df["Count"] / train.shape[0], name="Q")], axis=1)
        return f


    def getF(self, j, m, p, train_dict = None):
        def f(i):
            Qtrain = self.getQ(train_dict)(i)
            df = pd.DataFrame(self.train_set.groupby(by=["Class", self.data.features[j]])["Class"].agg("count")).rename(
                columns={"Class": "Count"})
            y = []
            for ((cl, _), count) in df["Count"].to_dict().items():
                y.append((count + 1 + m * p) / (Qtrain.at[cl, "Count"] + len(self.data.features) + m))
            print(y)
            #df['F'] = y
            #return df
            F = df.index.to_series().map(lambda t: (df["Count"][t] + 1 + m * p) /
                                                   (Qtrain.at[t[0], "Count"] + len(self.data.features) + m))
            return pd.concat([df, pd.Series(F, name = "F")], axis = 1)
        return f