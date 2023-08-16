import pandas as pd
import os


class MLData:
    def __init__(self, name, data_loc, columns, target_name, replace, classification):
        self.name = name
        self.data_loc = data_loc
        self.df = pd.read_csv(self.data_loc, header=None)
        self.columns = columns
        self.df.columns = self.columns
        self.target_name = target_name
        target_column = self.df.pop(self.target_name)
        self.features = self.df.columns
        self.df.insert(len(self.df.columns), "Target", target_column)
        self.replace = replace
        self.classification = classification

    def __str__(self):
        return self.name

    def one_hot(self):
        (features_numerical, features_categorical) = ([], [])
        features_categorical_ohe = []
        for f in self.features:
            try:
                self.df[f].apply(pd.to_numeric)
                features_numerical.append(f)
            except:
                features_categorical.append(f)
                categories = set(self.df[f])
                for cat in categories:
                    features_categorical_ohe.append(
                        "{}_{}".format(f, cat))
        self.features_numerical = features_numerical
        self.features_categorical = features_categorical
        one_hot_df = pd.get_dummies(self.df, columns=self.features_categorical)
        self.features_ohe = features_numerical + features_categorical_ohe
        target_column = one_hot_df.pop('Target')
        one_hot_df.insert(len(one_hot_df.columns), 'Target', target_column)
        self.df = one_hot_df

    def z_score_normalize(self):
        for col in self.features_ohe:
            std = self.df[col].std()  # computes standard deviation
            if std != 0:
                self.df[col] = (self.df[col] - self.df[col].mean()) / std  # column is normalized by z score

