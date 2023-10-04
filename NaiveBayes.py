import pandas as pd
import numpy as np
from copy import copy, deepcopy
import random
import math

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
        (p, j) = ([], 0)
        for i in range(r):
            p.append(list(range(j, j + q + 1)))
            j += q + 1
        for i in range(r, k):
            p.append(list(range(j, j + q)))
            j += q
        return p

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

    def getQ(self):
        df = pd.DataFrame(self.train_set.groupby(by = ["Class"])["Class"].agg('count')).rename(columns =
                                                                                               {"Class": "Count"})
        df["Q"] = df["Count"].apply(lambda x: x / self.train_set.shape[0])
        return df


    def getF(self, j, m, p, Qtrain=None):
        if Qtrain is None: Qtrain = self.getQ()
        df = pd.DataFrame(self.train_set.groupby(by=["Class", self.data.features[j]])["Class"].agg("count")).rename(
            columns={"Class": "Count"})
        y = []
        for ((cl, _), count) in df["Count"].to_dict().items():
            y.append((count + 1 + m * p) / (Qtrain.at[cl, "Count"] + len(self.data.features) + m))
        df['F'] = y
        return df