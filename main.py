from DataDictionary import DataDictionary

def f(i):
    if i == 1:
        DD = DataDictionary()
        data = DD.dataobject("Hardware")
        data.df.to_csv("data.csv")

if __name__ == '__main__':
    f(1)

