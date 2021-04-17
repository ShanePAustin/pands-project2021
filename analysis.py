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
def dataSummary():
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


    #create Summary.txt
    with open("Summary.txt", "w") as f:
        
        f.write(("Data Summary\n\n"))
        
        f.write(("Data Shape\n\n")+(str(shape)+('\n\n')))
        f.write(("Data Count\n\n")+(str(count)+('\n\n')))    
        f.write(("Data Count per Species\n\n")+(str(countSpecies)+('\n\n')))

        f.write(("Data Head per Species\n\n")+(str(head)+('\n\n'))) 

        f.write(("Data Summary \n\n")+(str(summary)+('\n\n')))
        f.write(("Data Summary (Setosa)\n\n")+(str(sumSeto)+('\n\n')))   
        f.write(("Data Summary (Versicolor)\n\n")+(str(sumVers)+('\n\n')))
        f.write(("Data Summary (Virginica)\n\n")+(str(sumVirg)+('\n\n')))

        f.write(("Data Correlation\n\n")+(str(correlation)))

###############################################################################################################################
#Histograms
def histograms():

    ax = plt.axes()
    plt.hist(df["Sepal Length"], color = "firebrick")
    plt.title("Sepal Length Histogram")
    plt.xlabel("Sepal Length")
    plt.grid(linestyle = "dashed", )
    ax.set_facecolor("lightgrey")
    plt.savefig("./plots/SLHist.png")
    plt.show()
    

    ax = plt.axes()
    plt.hist(df["Sepal Width"], color = "royalblue")
    plt.title("Sepal Width Histogram")
    plt.xlabel("Sepal Width")
    plt.grid(linestyle = "dashed", )
    ax.set_facecolor("lightgrey")    
    plt.savefig("./plots/SWHist.png")
    plt.show()

    ax = plt.axes()
    plt.hist(df["Petal Length"], color = "forestgreen")
    plt.title("Petal Length Histogram")
    plt.xlabel("Petal Length")
    plt.grid(linestyle = "dashed", )
    ax.set_facecolor("lightgrey")
    plt.savefig("./plots/PLHist.png")
    plt.show()
    

    ax = plt.axes()
    plt.hist(df["Petal Width"], color = "violet")
    plt.title("Petal Width Histogram")
    plt.xlabel("Petal Width")
    plt.grid(linestyle = "dashed", )
    ax.set_facecolor("lightgrey")
    plt.savefig("./plots/PWHist.png")
    plt.show()
    

##########################################################################
#Scatterplots
def scatterPlots():
    
    sns.FacetGrid(df,hue="Species", height=6).map(plt.scatter, "Sepal Length", "Sepal Width" ).add_legend()
    sns.FacetGrid(df,hue="Species", height=6).map(plt.scatter, "Petal Length", "Petal Width" ).add_legend()
    sns.FacetGrid(df,hue="Species", height=6).map(plt.scatter, "Sepal Length", "Petal Length").add_legend()
    sns.FacetGrid(df,hue="Species", height=6).map(plt.scatter, "Sepal Width" , "Petal Width" ).add_legend()

    sns.pairplot(df,hue="Species",height=3)
    plt.savefig("./plots/scatterPlot.png")
    plt.show()

##########################################################################
def distPlot():
    histplot, axes = plt.subplots(2,2, figsize=(10,10), sharex=False)
    sns.histplot( df["Sepal Length"] , kde = True, stat="density", linewidth=1, color="firebrick"  , ax=axes[0, 0])
    sns.histplot( df["Sepal Width" ] , kde = True, stat="density", linewidth=1, color="royalblue"  ,ax=axes[0, 1]) 
    sns.histplot( df["Petal Length"] , kde = True, stat="density", linewidth=1, color="forestgreen", ax=axes[1, 0]) 
    sns.histplot( df["Petal Width" ] , kde = True, stat="density", linewidth=1, color="violet"     , ax=axes[1, 1])
    plt.savefig("./plots/distPlot.png")
    plt.show()
##########################################################################
def stack():
    df.plot(kind='barh', stacked=True)  
    plt.xlabel('in cm')  
    plt.ylabel('Sample of 150 flowers') 
    plt.title('Stacked bar graph of Iris dataset')
    plt.savefig("./plots/stackPlot.png")
    plt.show() 

##########################################################################
def boxPlots():
    boxPlot, axes = plt.subplots(1,4, figsize=(16,8))
    sns.boxplot(x="Species", y="Sepal Length", hue="Species", data=df, palette="bright", ax=axes[0], dodge=False)
    sns.boxplot(x="Species", y="Sepal Width" , hue="Species", data=df, palette="bright", ax=axes[1], dodge=False)
    sns.boxplot(x="Species", y="Petal Length", hue="Species", data=df, palette="bright", ax=axes[2], dodge=False)
    sns.boxplot(x="Species", y="Petal Width" , hue="Species", data=df, palette="bright", ax=axes[3], dodge=False)
    plt.savefig("./plots/boxPlots.png")
    plt.show()

def violinPlots():
    violinplot, axes = plt.subplots(2,2, figsize=(10,10), sharex=False)
    sns.violinplot(x="Species", y="Sepal Length", hue="Species", data=df, palette="bright", ax=axes[0,0], dodge=False)
    sns.violinplot(x="Species", y="Sepal Width" , hue="Species", data=df, palette="bright", ax=axes[0,1], dodge=False)
    sns.violinplot(x="Species", y="Petal Length", hue="Species", data=df, palette="bright", ax=axes[1,0], dodge=False)
    sns.violinplot(x="Species", y="Petal Width" , hue="Species", data=df, palette="bright", ax=axes[1,1], dodge=False)
    plt.savefig("./plots/violinPlots.png")
    plt.show()

def heatMap():
    sns.heatmap(df.corr(),cmap="BuPu", annot=True)
    plt.savefig("./plots/hMap.png")
    plt.show()


#dataSummary()
#histograms()
scatterPlots()
distPlot()
boxPlots()
violinPlots()
stack()
heatMap()
