import pandas as pd
import numpy as np
from copy import copy, deepcopy
import random
import math
from functools import reduce
from itertools import product

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

    def training_test_dicts(self, df, partition=None):
        partition = self.partition(10) if partition is None else partition
        train_index = lambda i: reduce(lambda l1, l2: l1 + l2, partition[:i] + partition[i + 1:])
        test_dict = dict([(i, df.filter(items=partition[i], axis=0)) for i in range(len(partition))])
        train_dict = dict([(i, df.filter(items=train_index(i), axis=0)) for i in range(len(partition))])
        return (train_dict, test_dict)

    def getQ(self, df = None):
        df = self.data.df if df is None else df
        Q = pd.DataFrame(df.groupby(by=["Class"])["Class"].agg("count")).rename(columns={"Class": "Count"})
        return pd.concat([Q, pd.Series(Q["Count"] / df.shape[0], name="Q")], axis=1)


    def getF(self, m, p, df = None):
        #m is the number of pseudo-examples. p is the probability that the pseudo example occurs.
        df = self.data.df if df is None else df
        def g(j):
            Qframe = self.getQ(df)
            Fframe = pd.DataFrame(df.groupby(by=["Class", self.data.features[j]])["Class"].agg("count")).rename(
                columns={"Class": "Count"})
            Fcol = Fframe.index.to_series().map(lambda t: (Fframe["Count"][t] + 1 + m * p) /
                                                   (Qframe.at[t[0], "Count"] + len(self.data.features) + m))
            return pd.concat([Fframe, pd.Series(Fcol, name = "F")], axis = 1)
        return g

    def getFs(self, m, p, df = None):
        F_func = self.getF(m, p, df)
        return dict([(j, F_func(j)) for j in range(len(self.data.features))])

    def value(self, df):
        return lambda i: df.loc[i, self.data.features]

    def C(self, Qframe, Fframes):
        def f(cl, x):
            return reduce(lambda r, j: r * Fframes[j].to_dict()["F"].get((cl, x[j]), 0),
                          range(len(self.data.features)), Qframe.at[cl, "Q"])
        return f

    def predicted_class(self, Qframe, Fframes):
        C_func = self.C(Qframe, Fframes)
        def f(x):
            cl_prob = Qframe.index.map(lambda cl: (cl, C_func(cl, x)))
            return reduce(lambda t1, t2: t2 if t1[0] is None or t2[1] > t1[1] else t1, cl_prob, (None, None))[0]
        return f

