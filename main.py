from DataDictionary import DataDictionary
from NaiveBayes import NaiveBayes

def f(i):
    if i == 1:
        DD = DataDictionary()
        data = DD.dataobject("SoyBean")
        data.df.to_csv("data.csv")
        """
        NB = NaiveBayes(data)
        binned = NB.bin(5)
        binned.to_csv("binned_data.csv")
        noised = NB.getNoise()
        noised.to_csv("noisy_data.csv")
        print(NB.partition(10))
        NB.training_test_sets(1, NB.data.df)
        print(NB.test_set)
        print("===================================")
        print(NB.getQ())
        """

if __name__ == '__main__':
    f(1)

