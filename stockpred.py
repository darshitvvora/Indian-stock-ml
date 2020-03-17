import numpy as np
import pandas as pd
import tensorflow as tf
from datetime import date
from nsepy import get_history
import matplotlib.pyplot as plt
import functools
import math
from sklearn import preprocessing as pp

data = get_history(symbol="BIOCON", start=date(2017,1,1), end=date(2019,7,3), index = True)
print(data.head())
data.describe()