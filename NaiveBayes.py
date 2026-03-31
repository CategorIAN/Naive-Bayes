import pandas as pd
from functools import reduce, partial
from MLData import Dataset
from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class NaiveBayes:
    n: int # Bin Size
    alpha: float # Smoothing Parameter

    def bin_map(self, intervals):
        left = intervals.iloc[0].left
        right = intervals.iloc[-1].right

        def f(x):
            x = float(x)

            if x < left:
                return 0
            if x > right:
                return len(intervals) - 1

            for i, interval in enumerate(intervals):
                if x in interval:
                    return i

            raise ValueError(f"value {x} did not fit into intervals {list(intervals)}")
        return f

    def binned(self, data: Dataset) -> tuple[Dataset, dict[str, Callable]]:
        df = data.X

        def f(col: str) -> tuple[pd.Series, Callable[[object], object]]:
            try:
                colvalues = pd.to_numeric(df[col])
                nunique = colvalues.nunique(dropna=True)
                if nunique <= 1:
                    return df[col], lambda x: x
                bin_intervals = pd.qcut(colvalues, q=self.n, duplicates='drop').categories
                return self.bin_map(bin_intervals)
            except (ValueError, TypeError) as e:
                print(e)
                print("Not Numeric")
                return df[col], lambda x: x

        def g(mytuple, col):
            X, binner = mytuple
            cols, col_binner = f(col)
            return pd.concat([X, cols], axis=1), binner | {col: col_binner}

        new_data, binner = reduce(g, df.columns, (pd.DataFrame(), {}))
        return data.transform(new_data), binner

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
                df.groupby(["Class", feat], observed=False)
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

    def predict(self, Q, Fmap, binner, x):
        x_binned = {col: binner[col](v) for col, v in x.items()}
        class_prob = lambda cl: reduce(
            lambda r, feat: r * Fmap[feat].at[(cl, x_binned[feat]), "F"],
            x_binned.keys(),
            Q.at[cl, "Q"]
        )
        cl_probs = [(cl, class_prob(cl)) for cl in Q.index]
        return max(cl_probs, key=lambda t: t[1])[0]

    def __call__(self, data: Dataset):
        binned_data, binner = self.binned(data)
        Q = self.getQ(binned_data)
        Fmap = self.getFmap(Q, binned_data)
        return partial(self.predict, Q, Fmap, binner)



