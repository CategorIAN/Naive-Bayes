from NaiveBayes import NaiveBayes
from CrossValidation import CrossValidation as CV
from functools import reduce
import tail_recursive
from MLData import get_data


def f(i):
    if i == 1:
        my_sum = 0
        for i in range(0, 5):
            my_sum += i
        return my_sum
    if i == 2:
        my_sum = reduce(lambda x, y: x + y, range(0, 5), 0)
        return my_sum


def g(i):
    if i == 1:
        x = 0
        while x < 1000:
            x += 1
        return x
    if i == 2:
        @tail_recursive.tail_recursive
        def go(x):
            if x < 1000:
                return go.tail_call(x + 1)
            
            else:
                return x
        return go(0)


if __name__ == '__main__':
    get_data("soybean")
