import pandas as pd
from functools import reduce

class NaiveBayes:
    def __init__(self, data):
        self.data = data

    def binned(self, df, b):
        def f(col):
            try:
                colvalues = df[col].apply(pd.to_numeric)
                return pd.qcut(colvalues.rank(method="first"), q=b, labels=range(n))
            except:
                return df[col]
        return pd.DataFrame(dict([(col, f(col)) for col in df.columns]))

    def getQ(self, df):
        Q = pd.DataFrame(df.groupby(by=["Class"])["Class"].agg("count")).rename(columns={"Class": "Count"})
        return pd.concat([Q, pd.Series(Q["Count"] / df.shape[0], name="Q")], axis=1)

    def getF(self, df, p, m, Qframe):
        #m is the number of pseudo-examples. p is the probability that the pseudo example occurs.
        def g(j):
            Fframe = pd.DataFrame(df.groupby(by=["Class", self.data.features[j]])["Class"].agg("count")).rename(
                columns={"Class": "Count"})
            Fcol = Fframe.index.to_series().map(lambda t: (Fframe["Count"][t] + 1 + m * p) /
                                                   (Qframe.at[t[0], "Count"] + len(self.data.features) + m))
            return pd.concat([Fframe, pd.Series(Fcol, name = "F")], axis = 1)
        return g

    def getFs(self, df, p, m, Qframe):
        F_func = self.getF(df, p, m, Qframe)
        return dict([(j, F_func(j)) for j in range(len(self.data.features))])

    def class_prob(self, df, p, m, Qframe):
        Fframes = self.getFs(df, p, m, Qframe)
        def f(cl, x):
            return reduce(lambda r, j: r * Fframes[j].to_dict()["F"].get((cl, x[j]), 0),
                          range(len(self.data.features)), Qframe.at[cl, "Q"])
        return f

    def predicted_class(self, df, p, m):
        Qframe = self.getQ(df)
        class_prob_func = self.class_prob(df, p, m, Qframe)
        def f(x):
            cl_probs = Qframe.index.map(lambda cl: (cl, class_prob_func(cl, x)))
            return reduce(lambda t1, t2: t2 if t1[0] is None or t2[1] > t1[1] else t1, cl_probs, (None, None))[0]
        return f

    def value(self, df):
        return lambda i: df.loc[i, self.data.features]

    def target(self, i):
        return self.data.df.at[i, "Class"]

