import pandas as pd
from functools import reduce, partial
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

        def f(feat: str) -> pd.DataFrame:
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
        return {feat: f(feat) for feat in feats}

    def predict(self, Q, Fmap, x):
        class_prob = lambda cl, x: reduce(
            lambda r, feat: r * Fmap[feat].at[(cl, x[feat]), "F"],
            x.index,
            Q.at[cl, "Q"]
        )
        cl_probs = [(cl, class_prob(cl, x)) for cl in Q.index]
        return max(cl_probs, key=lambda t: t[1])[0]

    def __call__(self, data: Dataset):
        binned_data = self.binned(data)
        Q = self.getQ(binned_data)
        Fmap = self.getFmap(Q, binned_data)
        return partial(self.predict, Q, Fmap)



