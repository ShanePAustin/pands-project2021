#analysis.py
#A program for the Analysis of the Iris Data Set
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

###############################################################################################################################
#Data summary.txt
#summary production using describe()
summary = df.describe()
#Summary per flower type
sumSeto = df.loc[0:49].describe()
sumVers = df.loc[50:99].describe()
sumVirg = df.loc[100:49].describe()
#count the occurances of each flower type
count = df["Species"].value_counts()
#output the first 5 lines of data per species of flower
head = df.groupby("Species").head(5)
#output the correlation of the 4 attributes
correlation = df.corr()
'''
#create Summary.txt
with open("Summary.txt", "w") as f:
    
    f.write(("Data Summary\n\n")+(str(summary)+('\n\n')))
    f.write(("Data Summary (Setosa)\n\n")+(str(sumSeto)+('\n\n')))   
    f.write(("Data Summary (Versicolor)\n\n")+(str(sumVers)+('\n\n')))
    f.write(("Data Summary (Virginica)\n\n")+(str(sumVirg)+('\n\n')))
    f.write(("Species Count\n\n")+(str(count)+('\n\n')))
    f.write(("Data Head per Species\n\n")+(str(head)+('\n\n')))
    f.write(("Data Correlation\n\n")+(str(correlation)))
'''
###############################################################################################################################

sepallengthhist, axes = plt.subplots(figsize=(10,8))
df["Sepal Length"].plot(kind='hist',color='firebrick')
plt.xlabel("Sepal Length")
plt.grid(linestyle = "dashed")
plt.show()

sepalwidthhist, axes = plt.subplots(figsize=(10,8))
df["Sepal Width"].plot(kind='hist',color='royalblue')
plt.xlabel("Sepal Width")
plt.grid(linestyle = "dashed")
plt.show()

petallengthhist, axes = plt.subplots(figsize=(10,8))
df["Petal Length"].plot(kind='hist',color='forestgreen')
plt.xlabel("Petal Length")
plt.grid(linestyle = "dashed")
plt.show()

petallengthhist, axes = plt.subplots(figsize=(10,8))
df["Petal Width"].plot(kind='hist',color='violet')
plt.xlabel("Petal Width")
plt.grid(linestyle = "dashed")
plt.show()

Histogram, axes = plt.subplots(2,2, figsize=(10,10), sharex=False)
sns.histplot( df["Sepal Length"] , color="firebrick", ax=axes[0, 0])
sns.histplot( df["Sepal Width"] , color="royalblue",ax=axes[0, 1]) 
sns.histplot( df["Petal Length"] , color="forestgreen", ax=axes[1, 0]) 
sns.histplot( df["Petal Width"] , color="violet", ax=axes[1, 1])
#plt.show()

