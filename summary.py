#summary.py
#A test program that calls in the dataset and creates a summary of the data as a text file
#Author: Shane Austin


import csv
import pandas as pd

#import the csv
filename = "./data/iris.csv"

#reading in data and assigning column values
content = pd.read_csv(filename, header = None, names = ["Sepal Length", "Sepal Width", "Petal Length", "Petal Width", "Species"])

#summary production using describe()
summary = content.describe()
#count the occurances of each flower type
count = content["Species"].value_counts()
#output the first 5 lines of data per species of flower
head = content.groupby("Species").head(5)
#output the correlation of the 4 attributes
correlation = content.corr()

#create Summary.txt
with open("Summary.txt", "w") as f:
    
    f.write(str(summary)+('\n\n')+(str(count)+('\n\n')+(str(head)+('\n\n')+(str(correlation)))))

