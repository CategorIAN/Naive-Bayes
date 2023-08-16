import pandas as pd
import numpy as np
import os
from copy import copy

class MLData:
    def __init__(self, name, data_loc, columns, target_name, replace, classification):
        self.name = name
        self.data_loc = data_loc
        self.df = pd.read_csv(self.data_loc, header=None)
        self.columns = columns
        self.df.columns = self.columns
        self.target_name = target_name
        target_column = self.df.pop(self.target_name)
        self.features = list(self.df.columns)
        self.df.insert(len(self.df.columns), "Class", target_column)
        self.replace = replace
        self.classification = classification
        self.classes = pd.Index(list(set(self.df["Class"]))) if self.classification else None

    def __str__(self):
        return self.name


