import pandas as pd
from functools import reduce, partial
from MLData import Dataset
from dataclasses import dataclass
from typing import Callable, Union


@dataclass(frozen=True)
class NaiveBayes:
    """
    Discrete Naive Bayes classifier with numeric feature binning.

    Hyperparameters:
        n: Number of bins used for numeric features.
        alpha: Laplace/additive smoothing parameter.
    """
    n: int
    alpha: float

    @staticmethod
    def bin_func(intervals: pd.core.indexes.interval.IntervalIndex):
        """
        Create a function that maps numeric values to bin indices.

        Values outside the observed training range are clamped to the
        nearest edge bin so prediction can still proceed.
        """
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
        """
        Build a binning map for every feature.

        Numeric features are mapped to bin indices.
        Non-numeric features are left unchanged.

        Returns:
            {
                feature_name: (number_of_possible_values, mapping_function)
            }
        """
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
        """
        Apply learned binning functions to the feature matrix.

        The target column is preserved through Dataset.transform().
        """
        feature_data = data.X
        new_feature_data = pd.concat([feature_data[col].map(binner[col][1]) for col in feature_data.columns], axis=1)
        return data.transform(new_feature_data)

    @staticmethod
    def getQ(data: Dataset) -> pd.DataFrame:
        """
        Compute prior class probabilities.

        Returns a DataFrame indexed by class with:
            Count: number of training examples in that class
            Q: prior probability of that class
        """
        y = data.y
        counts = y.value_counts().rename("Count")
        return counts.to_frame().assign(Q=lambda df: df["Count"] / df["Count"].sum())

    @staticmethod
    def multi_count(data: Dataset) -> dict[str, dict[tuple[object, object], int]]:
        """
        Count occurrences of each (class, feature value) pair.

        These counts are later used to estimate conditional probabilities:
            P(feature value | class)
        """
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
        """
        Create a conditional probability lookup function.

        Uses Laplace smoothing:

            (count + alpha) / (class_count + alpha * K)

        where K is the number of possible values for the feature.
        """
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
        """
        Predict the class label for one observation.

        For each class, compute:

            P(class) * product P(feature value | class)

        Then return the class with the largest score.
        """
        cl_probs = [(cl, class_prob(cl)) for cl in Q.index]
        return max(cl_probs, key=lambda t: t[1])[0]

    def __call__(self, data: Dataset) -> Callable[[pd.Series], object]:
        """
        Train the model and return a classifier function.

        This creates the learned objects needed for prediction:
            - binner: feature transformation functions
            - Q: class priors
            - count_map: feature likelihood counts

        Returns:
            A function that maps one input row to a predicted class.
        """
        binner = self.bin_map(data)
        binned_data = self.binned(binner, data)
        Q = self.getQ(binned_data)
        count_map = self.multi_count(binned_data)
        return partial(self.predict, binner, Q, count_map)



