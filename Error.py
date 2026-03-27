import pandas as pd

def zero_one_loss(predicted: pd.Series, actual: pd.Series) -> float:
    return sum(predicted.to_numpy() != actual.to_numpy()).mean()