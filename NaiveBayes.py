import pandas as pd
from functools import reduce
from MLData import Dataset


def binned(df: pd.DataFrame, n: int) -> pd.DataFrame:
    def f(col):
        try:
            colvalues = pd.to_numeric(df[col])
            return pd.qcut(colvalues.rank(method="first"), q=n, labels=range(n))
        except (ValueError, TypeError):
            return df[col]
    return pd.DataFrame({col: f(col) for col in df.columns})


def merge(n: int):
    def f(data: Dataset) -> pd.DataFrame:
        binned_features = binned(data.X, n)
        return pd.concat([binned_features, data.y], axis=1)
    return f


def getQ(y: pd.Series) -> pd.DataFrame:
    counts = y.value_counts().rename("Count")
    return counts.to_frame().assign(Q=lambda df: df["Count"] / df["Count"].sum())


def getF(m: int):
    def f(df: pd.DataFrame, feats: [str], Q: pd.DataFrame) -> dict[int, pd.DataFrame]:
        def g(j):
            F = pd.DataFrame(df.groupby(by=["Class", feats[j]])["Class"].count()).rename(columns={"Class": "Count"})
            vals = F.index.to_series().map(lambda t: (F["Count"][t] + 1) / (Q.at[t[0], "Count"] + m))
            return pd.concat([F, pd.Series(vals, name = "F")], axis=1)
        return dict([(j, g(j)) for j in range(len(feats))])

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

