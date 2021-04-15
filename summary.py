#summary.py
#A test program that calls in the dataset and creates a summary of the data as a text file
#Author: Shane Austin


import csv
import pandas as pd

#import the csv
filename = "./data/iris.csv"

#reading in data and assigning column values
df = pd.read_csv(filename, header = None, names = ["Sepal Length", "Sepal Width", "Petal Length", "Petal Width", "Species"])

###############################################################################################################################
#Data summary.txt
#summary production using describe()
summary = df.describe().round(2)
#Summary per flower type
sumSeto = df.loc[0:49].describe().round(2)
sumVers = df.loc[50:99].describe().round(2)
sumVirg = df.loc[100:149].describe().round(2)
#count the occurances of each flower type
count = df["Species"].value_counts()
#output the first 5 lines of data per species of flower
head = df.groupby("Species").head(5)
#output the correlation of the 4 attributes
correlation = df.corr()

#create Summary.txt
with open("Summary.txt", "w") as f:
    
    f.write(("Data Summary\n\n")+(str(summary)+('\n\n')))
    f.write(("Data Summary (Setosa)\n\n")+(str(sumSeto)+('\n\n')))   
    f.write(("Data Summary (Versicolor)\n\n")+(str(sumVers)+('\n\n')))
    f.write(("Data Summary (Virginica)\n\n")+(str(sumVirg)+('\n\n')))
    f.write(("Species Count\n\n")+(str(count)+('\n\n')))
    f.write(("Data Head per Species\n\n")+(str(head)+('\n\n')))
    f.write(("Data Correlation\n\n")+(str(correlation)))

###############################################################################################################################

