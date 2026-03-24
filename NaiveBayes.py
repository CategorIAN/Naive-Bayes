import pandas as pd
from functools import reduce
from MLData import Dataset
from dataclasses import dataclass


@dataclass(frozen=True)
class NaiveBayes:
    n: int # Bin Size
    alpha: float # Smoothing Parameter

    def binned(self, df: pd.DataFrame) -> pd.DataFrame:
        def g(col: str) -> pd.Series:
            try:
                colvalues = pd.to_numeric(df[col])
                return pd.qcut(colvalues.rank(method="first"), q=self.n, labels=range(self.n))
            except (ValueError, TypeError):
                return df[col]
        return pd.DataFrame({col: g(col) for col in df.columns})

    @staticmethod
    def getQ(data: Dataset) -> pd.DataFrame:
        y = data.y
        counts = y.value_counts().rename("Count")
        return counts.to_frame().assign(Q=lambda df: df["Count"] / df["Count"].sum())

    def getFmap(self, data: Dataset, Q: pd.DataFrame) -> dict[int, pd.DataFrame]:
        df = data.df()
        feats = data.X.columns

        def f(j: int) -> pd.DataFrame:
            feat = feats[j]
            counts = (
                df.groupby(["Class", feat])
                  .size()
                  .rename("Count")
            )
            full_index = pd.MultiIndex.from_product(
                [data.classes, range(self.n)],
                names=["Class", feat]
            )
            F = counts.reindex(full_index, fill_value=0).to_frame()
            F["F"] = F.index.to_series().map(
                lambda t: (F.at[t, "Count"] + self.alpha) / (Q.at[t[0], "Count"] + self.alpha * self.n)
            )
            return F
        return {j: f(j) for j in range(len(feats))}

    def class_prob(self, Q: pd.DataFrame, Fmap: dict[int, pd.DataFrame]):
        return lambda x, cl: reduce(lambda r, j: r * Fmap[j][cl, x[j]], range(len(x)), Q.at[cl, "Q"])

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

