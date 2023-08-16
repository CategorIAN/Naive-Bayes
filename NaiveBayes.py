import pandas as pd
import numpy as np
from copy import copy, deepcopy
import random
import math

class NaiveBayes:
    def __init__(self, data):
        self.data = data
        self.seed = random.random()

    def bin(self, n):
        binned_df = copy(self.data.df)
        for col_name in self.data.features:
            try:
                binned_df[col_name] = binned_df[col_name].apply(pd.to_numeric)
                binned_df[col_name] = pd.qcut(binned_df[col_name].rank(method='first'), q=n, labels=np.arange(n) + 1)
            except:
                pass
        return binned_df

    def getNoise(self):
        df = deepcopy(self.data.df)
        random.seed(self.seed)
        noise_features = random.sample(self.data.features, k=math.ceil(len(self.data.features) * .1))
        for feature in noise_features:
            df[feature] = pd.Series(np.random.permutation(df[feature]))
        return pd.DataFrame(df)

    def partition(self, k):
        n = self.data.df.shape[0]
        (q, r) = (n // k, n % k)
        (p, j) = ([], 0)
        for i in range(r):
            p.append(list(range(j, j + q + 1)))
            j += q + 1
        for i in range(r, k):
            p.append(list(range(j, j + q)))
            j += q
        return p

    def training_test_sets(self, j, df, partition=None):
        if partition is None: partition = self.partition(10)
        train = []
        for i in range(len(partition)):
            if j != i:
                train += partition[i]
            else:
                test = partition[i]
        self.train_set = df.filter(items=train, axis=0)
        self.test_set = df.filter(items=test, axis=0)

    def getQ(self):
        df = pd.DataFrame(self.train_set.groupby(by = ["Class"])["Class"].agg('count')).rename(columns =
                                                                                               {"Class": "Count"})
        df["Q"] = df["Count"].apply(lambda x: x / self.train_set.shape[0])
        return df


    def getF(self, j, m, p, Qtrain=None):
        if Qtrain is None: Qtrain = self.getQ()
        df = pd.DataFrame(self.train_set.groupby(by=["Class", self.data.features[j]])["Class"].agg("count")).rename(
            columns={"Class": "Count"})
        y = []
        for ((cl, _), count) in df["Count"].to_dict().items():
            y.append((count + 1 + m * p) / (Qtrain.at[cl, "Count"] + len(self.data.features) + m))
        df['F'] = y
        return df