from functools import reduce as rd
import pandas as pd
import os
from itertools import product as prod
from functools import partial as pf
import time
from copy import copy

class CrossValidation:
    def __init__(self, data):
        self.data = data

