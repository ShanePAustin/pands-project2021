##### pands-project2021

# Iris Flower Data Set

### Programming and Scripting 52167

### Shane Austin G00318488

## Table of Contents

1. [Problem Statement](#problem-statement)
2. [Introduction](#introduction)
3. [Data Set](#data-set)
    1. [Data Summary](#data-summary)
4. [References](#references)

## Problem Statement 

*This project concerns the well-known Fisher’s Iris data set. You must research the data set and write documentation and code to investigate it. An online search for information on the data set will convince you that many people have investigated it previously. You are expected to be able to break this project into several smaller tasks that are easier to solve, and to plug these together after they have been completed.*

*You might do that for this project as follows:*
1. Research the data set online and write a summary about it in your README. 
2. Download the data set and add it to your repository. 
3. Write a program called analysis.py that: 
* outputs a summary of each variable to a single text file 
* saves a histogram of each variable to png files

## Introduction
The Iris flower dataset was introduced by the British statistician and biologist Ronald Fisher in 1936 in his paper (“The Use of Multiple Measurements in Taxonomic Problems”) [3] It focused on how to differentiate Iris species based on the shape and structure of their flowers. The dataset was originally collected by the botanist Edgar Anderson at the Gaspé Peninsula, Canada. Ronald Fisher was regarded by some as the single most important figure in 20th century statistics, and this dataset has since become a typical test case for many statistical classification techniques in the areas of statistics and machine learning.

The data set consists of 50 samples from each three species of Iris:

|Iris Setosa|Iris Versicolor|Iris Virginica|
|-----------|---------------|--------------|
|![alt text](https://github.com/ShanePAustin/pands-project2021/blob/main/Images/Iris_setosa.jpg "Iris Setosa")|![alt text](https://github.com/ShanePAustin/pands-project2021/blob/main/Images/Iris_versicolor.jpg "Iris Versicolor")|![alt text](https://github.com/ShanePAustin/pands-project2021/blob/main/Images/Iris_virginica.jpg "Iris Virginica")|


## Data Set

The data was found and downloaded form the UC Irvine Machine Learning Repository [4], it contains 4 attributes and 150 instances, split into 3 seperate classes (Iris species) of 50 each. A sample of the raw data is shown below:

![alt text](https://github.com/ShanePAustin/pands-project2021/blob/main/Images/RawData.png "Raw Data")

The four variables that were measured for each flower species are shown below:

1. Sepal Length (cm)
2. Sepal Width (cm)
3. Petal Length (cm)
4. Petal Width (cm)


### Data Summary

The data was imported into the program and the column names were defined:

``` python
filename = "./data/iris.csv"

df = pd.read_csv(filename, header = None, names = ["Sepal Length", "Sepal Width", "Petal Length", "Petal Width", "Species"])
```

With the data imported a summary was performed to assess the data integrity and discern the overall structure. To test data integrity 3 tests were performed to check for missing fields and data consistency. The tests performed were: 

1. Shape - This is a utility of NumPy to return the number of attributes and the length of each attribute.
2. Count - This expands on shape and shows the length of each attribute to check if no data is missing
3. Count per Species - This check the balance of the dataset to confirm each flower species has an equal length

The code and the ouput of those tests is shown here:

``` python
shape = df.shape

count = df.count()

countSpecies = df["Species"].value_counts()
```
![alt text](https://github.com/ShanePAustin/pands-project2021/blob/main/Images/DataCounts.png "Data Counts")


The image below shows the first ten lines of the dataset:
![alt text](https://github.com/ShanePAustin/pands-project2021/blob/main/Images/First10.png "First 10")

|       |Sepal Length | Sepal Width | Petal Length | Petal Width|
|-------|-------------|-------------|--------------|------------|
|count  |    150.0    |    150.0    |     150.0    |    150.0   |
|mean   |    5.843333 |    3.054000 |     3.758667 |    1.198667|
|std    |    0.828066 |    0.433594 |     1.764420 |    0.763161|
|min    |    4.300000 |    2.000000 |     1.000000 |    0.100000|
|25%    |    5.100000 |    2.800000 |     1.600000 |    0.300000|
|50%    |    5.800000 |    3.000000 |     4.350000 |    1.300000|
|75%    |    6.400000 |    3.300000 |     5.100000 |    1.800000|
|max    |    7.900000 |    4.400000 |     6.900000 |    2.500000|


```
Iris-versicolor    50  
Iris-setosa        50  
Iris-virginica     50  
Name: Flower, dtype: int64
```

## References

1) https://towardsdatascience.com/the-iris-dataset-a-little-bit-of-history-and-biology-fb4812f5a7b5

2) https://en.wikipedia.org/wiki/Iris_flower_data_set

3) https://digital.library.adelaide.edu.au/dspace/bitstream/2440/15227/1/138.pdf

4) https://archive.ics.uci.edu/ml/datasets/iris

5) https://rpubs.com/AjinkyaUC/Iris_DataSet

6) https://medium.com/@avulurivenkatasaireddy/exploratory-data-analysis-of-iris-data-set-using-python-823e54110d2d

7) https://www.researchgate.net/publication/344196553_Machine-learning_analysis_for_the_Iris_dataset

