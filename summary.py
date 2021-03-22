import csv
import pandas as pd

#import the csv
filename = "./data/iris.csv"

#reading in data and assigning column values
content = pd.read_csv(filename, header = None, names = ["Sepal Length", "Sepal Width", "Petal Length", "Petal Width", "Flower"])

#summary production using describe()
summary = content.describe()
#count the occurances of each flower type
count = content["Flower"].value_counts()

#create Summary.txt
with open("Summary.txt", "w") as f:
    
    f.write(str(summary))
    f.write('\n\n')
    f.write(str(count))