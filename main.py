from DataDictionary import DataDictionary
from NaiveBayes import NaiveBayes
from CrossValidation import CrossValidation as CV

def f(i):
    if i == 3:
        DD = DataDictionary()
        data = DD.dataobject("SoyBean")
        X = CV(data)
        X.test([11], [1], [100])



if __name__ == '__main__':
    f(3)

