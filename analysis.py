#analysis.py
#
#Author: Shane Austin


import csv
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

#import the csv
filename = "./data/iris.csv"

#reading in data and assigning column values
df = pd.read_csv(filename, header = None, names = ["Sepal Length", "Sepal Width", "Petal Length", "Petal Width", "Species"])

