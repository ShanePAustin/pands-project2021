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
    sumSeto = df[df["Species"] == "Iris-setosa"].describe().round(2)
    sumVers = df[df["Species"] == "Iris-versicolor"].describe().round(2)
    sumVirg = df[df["Species"] == "Iris-virginica"].describe().round(2)

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
#Histograms using matplotlib
#Function to create 4 Histograms for each of the measured variables and save the output to image file

def histograms():

    ax = plt.axes()
    plt.hist(df["Sepal Length"], color = "#810f7c", ec = "black")
    plt.title("Sepal Length Histogram")
    plt.xlabel("Sepal Length")
    ax.set_facecolor("lightgrey")
    plt.savefig("./plots/SLHist.png")
    plt.show()
    

    ax = plt.axes()
    plt.hist(df["Sepal Width"], color = "#8856a7", ec = "black")
    plt.title("Sepal Width Histogram")
    plt.xlabel("Sepal Width")
    ax.set_facecolor("lightgrey")    
    plt.savefig("./plots/SWHist.png")
    plt.show()

    ax = plt.axes()
    plt.hist(df["Petal Length"], color = "#8c96c6", ec = "black")
    plt.title("Petal Length Histogram")
    plt.xlabel("Petal Length")
    ax.set_facecolor("lightgrey")
    plt.savefig("./plots/PLHist.png")
    plt.show()
    

    ax = plt.axes()
    plt.hist(df["Petal Width"], color = "#b3cde3", ec = "black")
    plt.title("Petal Width Histogram")
    plt.xlabel("Petal Width")
    ax.set_facecolor("lightgrey")
    plt.savefig("./plots/PWHist.png")
    plt.show()
    

##########################################################################
#Scatterplots using Seaborn
#Function to create 4 Scatterplots seperated by Species, for each of the variable pairs and save the output to image file
#A seaborn pairplot to combine scatter and kernal density estimate

def scatterPlots():
    
    sns.FacetGrid(df,hue="Species", palette="BuPu_r", height=5).map(plt.scatter, "Sepal Length", "Sepal Width" ).add_legend()
    plt.title("Sepal Length / Sepal Width Scatter Plot")
    plt.subplots_adjust(top=0.9)
    plt.savefig("./plots/SLSWscatterPlot.png")

    sns.FacetGrid(df,hue="Species", palette="BuPu_r", height=5).map(plt.scatter, "Petal Length", "Petal Width" ).add_legend() 
    plt.title("Petal Length / Petal Width Scatter Plot")     
    plt.subplots_adjust(top=0.9) 
    plt.savefig("./plots/PLPWscatterPlot.png")

    sns.FacetGrid(df,hue="Species", palette="BuPu_r", height=5).map(plt.scatter, "Sepal Length", "Petal Length").add_legend()
    plt.title("Sepal Length / Petal Length Scatter Plot")
    plt.subplots_adjust(top=0.9)
    plt.savefig("./plots/SLPLscatterPlot.png")  
  
    sns.FacetGrid(df,hue="Species", palette="BuPu_r", height=5).map(plt.scatter, "Sepal Width" , "Petal Width" ).add_legend()
    plt.title("Sepal Width / Petal Width Scatter Plot")
    plt.subplots_adjust(top=0.9)
    plt.savefig("./plots/SWPWscatterPlot.png")

    sns.pairplot(df,hue="Species", palette="BuPu_r", height=3)
    plt.savefig("./plots/scatterPlot.png")

    plt.show()

##########################################################################
#Stacked histograms showing the values of each species. Combined into one output png

def distPlot():
    distplot, axes = plt.subplots(2,2, figsize=(10,10), sharex=False)
    sns.histplot( x="Sepal Length", hue="Species", data = df, palette="BuPu_r", ax=axes[0,0], multiple = "stack")
    sns.histplot( x="Sepal Width",  hue="Species", data = df, palette="BuPu_r", ax=axes[0,1], multiple = "stack")
    sns.histplot( x="Petal Length", hue="Species", data = df, palette="BuPu_r", ax=axes[1,0], multiple = "stack")
    sns.histplot( x="Petal Width",  hue="Species", data = df, palette="BuPu_r", ax=axes[1,1], multiple = "stack")
    plt.suptitle("Combined Histograms Seperated by Species")
    plt.savefig("./plots/distPlot.png")
    plt.show()

##########################################################################
#Seaborn Box Plots of each sppecies by measured variable. Combined into one output png

def boxPlots():
    boxPlot, axes = plt.subplots(1,4, figsize=(16,8))
    sns.boxplot(x="Species", y="Sepal Length", hue="Species", data=df, palette="BuPu_r", ax=axes[0], dodge=False)    
    sns.boxplot(x="Species", y="Sepal Width" , hue="Species", data=df, palette="BuPu_r", ax=axes[1], dodge=False)
    sns.boxplot(x="Species", y="Petal Length", hue="Species", data=df, palette="BuPu_r", ax=axes[2], dodge=False)
    sns.boxplot(x="Species", y="Petal Width" , hue="Species", data=df, palette="BuPu_r", ax=axes[3], dodge=False)
    plt.suptitle("Box Plots")
    plt.savefig("./plots/boxPlots.png")
    plt.show()

##########################################################################
#Seaborn Violin Plots of each sppecies by measured variable. Combined into one output png

def violinPlots():
    violinplot, axes = plt.subplots(2,2, figsize=(10,10), sharex=False)
    sns.violinplot(x="Species", y="Sepal Length", hue="Species", data=df, palette="BuPu_r", ax=axes[0,0], dodge=False)
    sns.violinplot(x="Species", y="Sepal Width" , hue="Species", data=df, palette="BuPu_r", ax=axes[0,1], dodge=False)
    sns.violinplot(x="Species", y="Petal Length", hue="Species", data=df, palette="BuPu_r", ax=axes[1,0], dodge=False)
    sns.violinplot(x="Species", y="Petal Width" , hue="Species", data=df, palette="BuPu_r", ax=axes[1,1], dodge=False)
    plt.suptitle("Violin Plots")
    plt.savefig("./plots/violinPlots.png")
    plt.show()

##########################################################################
#Seaborn Heat Map of the correlation table

def heatMap():
    sns.heatmap(df.corr(),cmap="BuPu", annot=True)
    plt.title("Correlation Heat Map")
    plt.savefig("./plots/hMap.png")
    plt.show()

##########################################################################
#Commands to run each Function

#dataSummary()
#histograms()
scatterPlots()
#distPlot()
#boxPlots()
#violinPlots()
#heatMap()
