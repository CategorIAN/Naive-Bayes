import pandas as pd
from functools import reduce, partial
from MLData import Dataset
from dataclasses import dataclass
from typing import Callable, Union


@dataclass(frozen=True)
class NaiveBayes:
    n: int # Bin Size
    alpha: float # Smoothing Parameter

    @staticmethod
    def bin_func(intervals: pd.core.indexes.interval.IntervalIndex):
        left = intervals[0].left
        right = intervals[-1].right

        def f(x: Union[int, float]):
            x = float(x)
            if x < left:
                return 0
            if x > right:
                return len(intervals) - 1
            for i, interval in enumerate(intervals):
                if x in interval:
                    return i
        return f

    def bin_map(self, data: Dataset) -> dict[str, tuple[int, Callable]]:
        df = data.X

        def f(col: str) -> tuple[int, Callable[[object], object]]:
            vals = df[col]
            nunique = vals.nunique(dropna=True)
            try:
                if nunique <= 1:
                    raise ValueError
                numvalues = pd.to_numeric(vals)
                bin_intervals = pd.cut(numvalues, bins=self.n).cat.categories
                return nunique, self.bin_func(bin_intervals)
            except (ValueError, TypeError) as e:
                return nunique, lambda x: x

        return {col: f(col) for col in df.columns}

    @staticmethod
    def binned(binner: dict[str, tuple[int, Callable]], data: Dataset) -> Dataset:
        feature_data = data.X
        new_feature_data = pd.concat([feature_data[col].map(binner[col][1]) for col in feature_data.columns], axis=1)
        return data.transform(new_feature_data)

    @staticmethod
    def getQ(data: Dataset) -> pd.DataFrame:
        y = data.y
        counts = y.value_counts().rename("Count")
        return counts.to_frame().assign(Q=lambda df: df["Count"] / df["Count"].sum())

    @staticmethod
    def multi_count(data: Dataset) -> dict[str, dict[tuple[object, object], int]]:
        df = data.df()
        feats = data.X.columns

        def f(feat: str) -> dict[tuple[object, object], int]:
            return dict(df.groupby(["Class", feat], observed=False).size().rename("Count"))
        return {feat: f(feat) for feat in feats}

    def prob_func(self,
                  binner: dict[str, tuple[int, Callable]],
                  Q: pd.DataFrame,
                  count_map: dict[str, dict[tuple[object, object], int]]
                  ) -> Callable:
        def f(feat: str, cl: str, val: object) -> float:
            my_map = count_map[feat]
            numerator = my_map.get((cl, val), 0) + self.alpha
            denominator = Q.at[cl, "Count"] + self.alpha * binner[feat][0]
            return numerator / denominator
        return f

    def predict(self,
                binner: dict[str, tuple[int, Callable]],
                Q: pd.DataFrame,
                count_map: dict[str, dict[tuple[object, object], int]],
                x: pd.Series
                ) -> object:
        my_prob_func = self.prob_func(binner, Q, count_map)
        x_binned = pd.Series({col: binner[str(col)][1](v) for col, v in x.items()})
        class_prob = lambda cl: reduce(
            lambda r, feat: r * my_prob_func(feat, cl, x_binned[feat]),
            x_binned.index,
            Q.at[cl, "Q"]
        )
        cl_probs = [(cl, class_prob(cl)) for cl in Q.index]
        return max(cl_probs, key=lambda t: t[1])[0]

    def __call__(self, data: Dataset) -> Callable[[pd.Series], object]:
        binner = self.bin_map(data)
        binned_data = self.binned(binner, data)
        Q = self.getQ(binned_data)
        count_map = self.multi_count(binned_data)
        return partial(self.predict, binner, Q, count_map)



