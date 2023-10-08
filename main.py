from DataDictionary import DataDictionary
from NaiveBayes import NaiveBayes

def f(i):
    if i == 1:
        DD = DataDictionary()
        data = DD.dataobject("SoyBean")
        data.df.to_csv("data.csv")
        NB = NaiveBayes(data)
        binned = NB.binned(5)
        binned.to_csv("binned_data.csv")
        noised = NB.noised()
        noised.to_csv("noisy_data.csv")
        print(NB.partition(10))
        traind, testd = NB.training_test_dicts(binned)
        #print(traind[0])
        print("===================================")
        Fs = NB.getFs(5, 0.03, traind[0])
        Q = NB.getQ(traind[0])
        for j in Fs.keys():
            print(j)
            print("-------")
            print(Fs[j])
        x = NB.value(binned)(5)
        print(x)
        print(NB.predicted_class(Q, Fs)(x))
    if i == 2:
        DD = DataDictionary()
        data = DD.dataobject("SoyBean")
        NB = NaiveBayes(data)
        binned = NB.binned(5)
        traind, testd = NB.training_test_dicts(binned)
        Fs = NB.getFs(5, 0.03, traind[0])
        Q = NB.getQ(traind[0])
        x = NB.value(binned)(5)
        print(NB.predicted_class(Q, Fs)(x))

if __name__ == '__main__':
    f(2)

