import pandas as pd
from functools import reduce
from MLData import Dataset
from dataclasses import dataclass


@dataclass(frozen=True)
class NaiveBayes:
    n: int # Bin Size
    alpha: float # Smoothing Parameter


    def binned(self, data: Dataset) -> Dataset:
        def g(df: pd.DataFrame, col: str) -> pd.Series:
            try:
                colvalues = pd.to_numeric(df[col])
                return pd.qcut(colvalues.rank(method="first"), q=self.n, labels=range(self.n))
            except (ValueError, TypeError):
                return df[col]

        return data.map(lambda df: pd.DataFrame({col: g(df, col) for col in df.columns}))

    @staticmethod
    def getQ(data: Dataset) -> pd.DataFrame:
        y = data.y
        counts = y.value_counts().rename("Count")
        return counts.to_frame().assign(Q=lambda df: df["Count"] / df["Count"].sum())

    def getFmap(self, Q: pd.DataFrame, data: Dataset) -> dict[int, pd.DataFrame]:
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

    def predicted_class(self, Q: pd.DataFrame, Fmap: dict[int, pd.DataFrame]):
        class_prob = lambda cl, x: reduce(lambda r, j: r * Fmap[j][cl, x[j]], range(len(x)), Q.at[cl, "Q"])

        def f(x):
            cl_probs = [(cl, class_prob(cl, x)) for cl in Q.index]
            return reduce(lambda t1, t2: t2 if t1[0] is t2[1] > t1[1] else t1, cl_probs[1:], cl_probs[0])[0]
        return f


