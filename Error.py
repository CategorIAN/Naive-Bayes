import pandas as pd
from MLData import Dataset
from NaiveBayes import NaiveBayes


def zero_one_loss(predicted: pd.Series, actual: pd.Series) -> float:
    return (predicted.to_numpy() != actual.to_numpy()).mean()


def error(model: NaiveBayes, train: Dataset, test: Dataset) -> float:
    classifier = model(train)
    predicted = test.X.apply(classifier, axis=1)
    actual = test.y
    return zero_one_loss(predicted, actual)


def predict(model: NaiveBayes, train: Dataset, test: Dataset) -> float:
    classifier = model(train)
    predicted = test.X.apply(classifier, axis=1)
    actual = test.y
    return zero_one_loss(predicted, actual)