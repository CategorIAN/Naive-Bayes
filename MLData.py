import pandas as pd
from dataclasses import dataclass
from pathlib import Path
import yaml


@dataclass(frozen=True)
class Dataset:
    name: str
    X: pd.DataFrame
    y: pd.Series
    task_type: str
    classes: pd.Index | None = None

    def df(self) -> pd.DataFrame:
        return pd.concat([self.X, self.y], axis=1)

    def map(self, f):
        return Dataset(
            name=self.name,
            X=f(self.X),
            y=self.y,
            task_type=self.task_type,
            classes=self.classes,
        )

    def filter(self, index):
        return Dataset(
            name=self.name,
            X=self.X.filter(items=index, axis=0),
            y=self.y.filter(items=index, axis=0),
            task_type=self.task_type,
            classes=self.classes,
        )


def load_data(name: str) -> pd.DataFrame:
    return pd.read_csv(Path("data_raw") / f"{name}.csv", header=None)


def load_meta(name: str) -> dict:
    path = Path("metadata") / f"{name}.yaml"
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def split_data(df: pd.DataFrame, target: str) -> tuple[pd.DataFrame, pd.Series]:
    X = df.drop(columns = [target])
    y = df[target].rename("Class")
    return X, y


def get_data(name: str) -> Dataset:
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



