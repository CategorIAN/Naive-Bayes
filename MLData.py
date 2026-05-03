import pandas as pd
from dataclasses import dataclass
from pathlib import Path
import yaml
from typing import Self


@dataclass(frozen=True)
class Dataset:
    """
    Lightweight container for a supervised learning dataset.

    X contains the feature columns.
    y contains the target column, renamed to 'Class'.
    task_type records whether the dataset is classification/regression.
    classes stores known class labels for classification datasets.
    """
    name: str
    X: pd.DataFrame
    y: pd.Series
    task_type: str
    classes: pd.Index | None = None

    def __str__(self):
        """Return a readable preview of the dataset."""
        pd.set_option("display.max_columns", None)   # show all columns
        return f"Dataset: {self.name}\n{10*'-'}\n{self.df().head()}\n{10*'-'}"

    def df(self) -> pd.DataFrame:
        """Return features and target as one combined DataFrame."""
        return pd.concat([self.X, self.y], axis=1)

    def transform(self, X: pd.DataFrame) -> Self:
        """
        Return a new Dataset with transformed features.

        The target, task type, and class labels are preserved.
        This supports functional-style transformations without mutating
        the original dataset.
        """
        return Dataset(
            name=self.name,
            X=X,
            y=self.y,
            task_type=self.task_type,
            classes=self.classes,
        )

    def filter(self, index: list[object]) -> Self:
        """
        Return a new Dataset restricted to selected row labels.

        Used for train/test splits during cross-validation.
        """
        return Dataset(
            name=self.name,
            X=self.X.filter(items=index, axis=0),
            y=self.y.filter(items=index, axis=0),
            task_type=self.task_type,
            classes=self.classes,
        )


def load_data(name: str) -> pd.DataFrame:
    """Load a raw CSV dataset by name from the data_raw folder."""
    return pd.read_csv(Path("data_raw") / f"{name}.csv", header=None)


def load_meta(name: str) -> dict:
    """Load dataset metadata from the metadata folder."""
    path = Path("metadata") / f"{name}.yaml"
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def split_data(df: pd.DataFrame, target: str) -> tuple[pd.DataFrame, pd.Series]:
    """Split a DataFrame into feature matrix X and target vector y."""
    X = df.drop(columns = [target])
    y = df[target].rename("Class")
    return X, y


def get_data(name: str) -> Dataset:
    """
    Load, label, clean, save, and package a dataset.

    This function connects the raw CSV file with its YAML metadata:
    - assigns column names
    - fills missing values if configured
    - saves a processed CSV for display in the web app
    - splits features and target
    - records class labels for classification tasks
    """
    data = load_data(name)
    meta = load_meta(name)
    data = data.set_axis(meta["columns"], axis=1)
    if meta["replace_value"] in meta:
        data = data.fillna(meta["replace_value"])
    data.to_csv(f"data_processed/{name}.csv", index=False)
    X, y = split_data(data, meta["target_name"])
    classes = pd.Index(y.unique()) if meta["task_type"] == "classification" else None

    return Dataset(
        name=name,
        X=X,
        y=y,
        task_type=meta["task_type"],
        classes=classes
    )



