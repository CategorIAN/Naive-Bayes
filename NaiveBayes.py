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

    def binned(self, df, n):
        def f(col):
            try:
                colvalues = df[col].apply(pd.to_numeric)
                return pd.qcut(colvalues.rank(method="first"), q=n, labels=range(n))
            except:
                return df[col]
        return pd.DataFrame(dict([(col, f(col)) for col in df.columns]))

    def noised(self):
        random.seed(self.seed)
        noise_features = random.sample(self.data.features, k=math.ceil(len(self.data.features) * .1))
        def f(col):
            return pd.Series(np.random.permutation(self.data.df[col])) if col in noise_features else self.data.df[col]
        return pd.DataFrame(dict([(col, f(col)) for col in self.data.df.columns]))

    def getQ(self, df = None):
        df = self.data.df if df is None else df
        Q = pd.DataFrame(df.groupby(by=["Class"])["Class"].agg("count")).rename(columns={"Class": "Count"})
        return pd.concat([Q, pd.Series(Q["Count"] / df.shape[0], name="Q")], axis=1)

    def getF(self, m, p, df = None):
        #m is the number of pseudo-examples. p is the probability that the pseudo example occurs.
        df = self.data.df if df is None else df
        Qframe = self.getQ(df)
        def g(j):
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

    def target(self, i):
        return self.data.df.at[i, "Class"]

