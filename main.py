from functools import reduce
from MLData import get_data
from NaiveBayes import NaiveBayes
from CrossValidation import CrossValidation


def f():
    data = get_data("soybean")
    nb = NaiveBayes(n=5, alpha=1)
    classifier = nb(data)
    print(classifier(data.X.iloc[1]))

def g():
    data = get_data("soybean")
    cv = CrossValidation(data, bin_sizes=[5], alphas=[1])
    cv.makePrediction(5, 1, 0)(0)


def h():
    data = get_data("soybean")
    cv = CrossValidation(data, bin_sizes=[1], alphas=[1])
    cv.predict()



if __name__ == '__main__':
    h()
