from functools import reduce
from MLData import get_data
from NaiveBayes import NaiveBayes
from CrossValidation import CrossValidation


def f():
    data = get_data("breast-cancer")
    print(data)

def g():
    data = get_data("soybean")
    cv = CrossValidation(data, bin_sizes=[5], alphas=[1])
    cv.makePrediction(5, 1, 0)(0)


def h():
    bin_sizes = [2, 3, 4, 5, 6, 7, 8, 9, 10]
    alphas = [0.01, 0.1, 0.5, 1, 2, 5, 10]
    data = get_data("breast-cancer")
    cv = CrossValidation(data, bin_sizes=bin_sizes, alphas=alphas)
    cv.predict()


if __name__ == '__main__':
    h()
