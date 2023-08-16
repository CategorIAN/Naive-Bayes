from MLData import MLData
import pandas as pd
from functools import partial as pf

class DataDictionary:
    def __init__(self):
        self.datanames = ["Hardware", "SoyBean"]

    def dataobject(self, name):
        data = MLData(*self.metadata(name))
        data.one_hot()
        data.z_score_normalize()
        data.classes = pd.Index(list(set(data.df['Target']))) if data.classification else None
        return data

    def metadata(self, name):
        if name == "Hardware": return self.hardware()
        if name == "SoyBean": return self.soybean()

    def hardware(self):
        name = "Hardware"
        file = 'raw_data/machine.csv'
        columns = [   "Vendor Name",  # For Computer Hardware
            "Model Name",
            "MYCT",
            "MMIN",
            "MMAX",
            "CACH",
            "CHMIN",
            "CHMAX",
            "PRP",  #Target
            "ERP"
        ]
        target_name = 'PRP'
        replace = None
        classification = False
        return (name, file, columns, target_name, replace, classification)

    def soybean(self):
        name = "SoyBean"
        file = 'raw_data/soybean-small.csv'
        columns =  ['Date',  # For Soy Bean
         'Plant-Stand',
         'Precip',
         'Temp',
         'Hail',
         'Crop-Hist',
         'Area-Damaged',
         'Severity',
         'Seed-TMT',
         'Germination',
         'Plant-Growth',
         'Leaves',
         'Leafspots-Halo',
         'Leafspots-Marg',
         'Leafspot-Size',
         'Leaf-Shread',
         'Leaf-Malf',
         'Leaf-Mild',
         'Stem',
         'Lodging',
         'Stem-Cankers',
         'Canker-Lesion',
         'Fruiting-Bodies',
         'External Decay',
         'Mycelium',
         'Int-Discolor',
         'Sclerotia',
         'Fruit-Pods',
         'Fruit Spots',
         'Seed',
         'Mold-Growth',
         'Seed-Discolor',
         'Seed-Size',
         'Shriveling',
         'Roots',
         'Class'  #Target
         ]
        replace = None
        target_name = 'Class'
        classification = True
        return (name, file, columns, target_name, replace, classification)
