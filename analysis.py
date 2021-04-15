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

shape = df.shape
count = df.count()

#count the occurances of each flower type
countSpecies = df["Species"].value_counts()

#output the first 5 lines of data per species of flower
head = df.groupby("Species").head(5)

#summary production using describe()
summary = df.describe().round(2)
#Summary per flower type
sumSeto = df.loc[0:49].describe().round(2)
sumVers = df.loc[50:99].describe().round(2)
sumVirg = df.loc[100:149].describe().round(2)

#output the correlation of the 4 attributes
correlation = df.corr()

'''
#create Summary.txt
with open("Summary.txt", "w") as f:
    
    f.write(("Data Summary"\n\n)
    
    f.write(("Data Shape\n\n")+(str(shape)+('\n\n')))
    f.write(("Data Count\n\n")+(str(count)+('\n\n')))    
    f.write(("Data Count per Species\n\n")+(str(countSpecies)+('\n\n')))

    f.write(("Data Head per Species\n\n")+(str(head)+('\n\n'))) 

    f.write(("Data Summary \n\n")+(str(summary)+('\n\n')))
    f.write(("Data Summary (Setosa)\n\n")+(str(sumSeto)+('\n\n')))   
    f.write(("Data Summary (Versicolor)\n\n")+(str(sumVers)+('\n\n')))
    f.write(("Data Summary (Virginica)\n\n")+(str(sumVirg)+('\n\n')))

    f.write(("Data Correlation\n\n")+(str(correlation)))
'''
###############################################################################################################################

ax = plt.axes()
plt.hist(df["Sepal Length"], color = "firebrick")
plt.title("Sepal Length Histogram")
plt.xlabel("Sepal Length")
plt.grid(linestyle = "dashed", )
ax.set_facecolor("lightgrey")
plt.show()

ax = plt.axes()
plt.hist(df["Sepal Width"], color = "royalblue")
plt.title("Sepal Width Histogram")
plt.xlabel("Sepal Width")
plt.grid(linestyle = "dashed", )
ax.set_facecolor("lightgrey")
plt.show()

ax = plt.axes()
plt.hist(df["Petal Length"], color = "forestgreen")
plt.title("Petal Length Histogram")
plt.xlabel("Petal Length")
plt.grid(linestyle = "dashed", )
ax.set_facecolor("lightgrey")
plt.show()

ax = plt.axes()
plt.hist(df["Petal Width"], color = "violet")
plt.title("Petal Width Histogram")
plt.xlabel("Petal Width")
plt.grid(linestyle = "dashed", )
ax.set_facecolor("lightgrey")
plt.show()
'''

Histogram, axes = plt.subplots(2,2, figsize=(10,10), sharex=False)
sns.histplot( df["Sepal Length"] , color="firebrick", ax=axes[0, 0])
sns.histplot( df["Sepal Width"] , color="royalblue",ax=axes[0, 1]) 
sns.histplot( df["Petal Length"] , color="forestgreen", ax=axes[1, 0]) 
sns.histplot( df["Petal Width"] , color="violet", ax=axes[1, 1])
#plt.show()

Distplot, axes = plt.subplots(2,2, figsize=(10,10), sharex=False)
sns.distplot( df["Sepal Length"] , color="firebrick", ax=axes[0, 0])
sns.distplot( df["Sepal Width"] , color="royalblue",ax=axes[0, 1]) 
sns.distplot( df["Petal Length"] , color="forestgreen", ax=axes[1, 0]) 
sns.distplot( df["Petal Width"] , color="violet", ax=axes[1, 1])
plt.show()
#plt.savefig("Histogram.png") 
'''

df.plot(kind='barh', stacked=True)  
plt.xlabel('in cm')  
plt.ylabel('Sample of 150 flowers') 
plt.title('Stacked bar graph of Iris dataset')
plt.show() 

