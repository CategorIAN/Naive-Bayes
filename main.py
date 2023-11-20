from DataDictionary import DataDictionary
from NaiveBayes import NaiveBayes
from CrossValidation import CrossValidation as CV

def f(i):
    if i == 3:
        DD = DataDictionary()
        data = DD.dataobject("SoyBean")
        print(len(data.df.columns))
        #X = CV(data)
        #print(X.best_params([11, 50], [.001, 1], [24, 100]))



if __name__ == '__main__':
    f(3)

