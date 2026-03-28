from functools import cached_property
import pandas as pd
from NaiveBayes import NaiveBayes
from dataclasses import dataclass
from MLData import Dataset
from collections.abc import Sequence
import os
import django
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "django-project.settings")
django.setup()
from myapp.models import Prediction

@dataclass(frozen=True)
class CrossValidation:
    data: Dataset
    bin_sizes: Sequence[int]
    alphas: Sequence[float]

    def partition(self) -> list[list[int]]:
        n = self.data.X.shape[0]
        (q, r) = (n // 10, n % 10)

        def f(i, j, p):
            if i == 10:
                return p
            else:
                return f(i + 1, j + q + int(i < r), p + [list(range(j, j + q + int(i < r)))])
        return f(0, 0, [])

    def _train_test_dict(self) -> dict[int, tuple[Dataset, Dataset]]:
        parts = self.partition()
        train = lambda i: self.data.filter([idx for j, fold in enumerate(parts) if j != i for idx in fold])
        test = lambda i: self.data.filter(parts[i])
        return {i: (train(i), test(i)) for i in range(10)}

    @cached_property
    def train_test_dict(self):
        return self._train_test_dict()

    @cached_property
    def prediction_index(self):
        return pd.Series([
            (bin_size, alpha, i, j)
            for bin_size in self.bin_sizes
            for alpha in self.alphas
            for i in range(10)
            for j in range(self.train_test_dict[i][1].X.shape[0])
        ])

    def makePrediction(self, bin_size, alpha, i):
        model = NaiveBayes(bin_size, alpha)
        train, test = self.train_test_dict[i]
        classifier = model(train)
        def f(j):
            predicted = classifier(test.X.iloc[j])
            actual = test.y.iloc[j]
            Prediction.objects.get_or_create(
                bin_size = bin_size,
                alpha = alpha,
                test_set_index = i,
                row_index = j,
                defaults={
                    "predicted": predicted,
                    "actual": actual
                }
            )
        return f

    def first_missing_prediction(self):
        existing = set(Prediction.objects.values_list("bin_size", "alpha", "test_set_index", "row_index"))

        for k, v in self.prediction_index.items():
            if v not in existing:
                return k

        return None

    def predict(self):
        k = self.first_missing_prediction()
        if k is None:
            return None

        prev_key = None
        predict_fn = None

        while k < len(self.prediction_index):
            bin_size, alpha, i, j = self.prediction_index[k]
            key = (bin_size, alpha, i)

            if predict_fn is None or key != prev_key:
                predict_fn = self.makePrediction(bin_size, alpha, i)
                prev_key = key

            predict_fn(j)
            k += 1



