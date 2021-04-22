##### pands-project2021

# Iris Flower Data Set

### Programming and Scripting 52167

### Shane Austin G00318488

## Table of Contents

1. [Problem Statement](#1-problem-statement)
2. [Introduction](#2-introduction)
    1. [Libraries](#2.1-libraries)
3. [Data Set](#3-data-set)
    1. [Data Summary](#3.1-data-summary)
        1. [Data Integrity](#3.1.1-data-integrity)
        2. [Data Statistics](#3.1.2-data-statistics)
4. [Data Analysis](#4-data-analysis)
    1. [Histograms](#4.1-histograms)
4. [References](#references)

## 1 Problem Statement 

*This project concerns the well-known Fisher’s Iris data set. You must research the data set and write documentation and code to investigate it. An online search for information on the data set will convince you that many people have investigated it previously. You are expected to be able to break this project into several smaller tasks that are easier to solve, and to plug these together after they have been completed.*

*You might do that for this project as follows:*
1. Research the data set online and write a summary about it in your README. 
2. Download the data set and add it to your repository. 
3. Write a program called analysis.py that: 
* outputs a summary of each variable to a single text file 
* saves a histogram of each variable to png files

## 2 Introduction
The Iris flower dataset was introduced by the British statistician and biologist Ronald Fisher in 1936 in his paper (“The Use of Multiple Measurements in Taxonomic Problems”) [3] It focused on how to differentiate Iris species based on the shape and structure of their flowers. The dataset was originally collected by the botanist Edgar Anderson at the Gaspé Peninsula, Canada. Ronald Fisher was regarded by some as the single most important figure in 20th century statistics, and this dataset has since become a typical test case for many statistical classification techniques in the areas of statistics and machine learning.

The data set consists of 50 samples from each three species of Iris:

|Iris Setosa|Iris Versicolor|Iris Virginica|
|-----------|---------------|--------------|
|![alt text](https://github.com/ShanePAustin/pands-project2021/blob/main/Images/Iris_setosa.jpg "Iris Setosa")|![alt text](https://github.com/ShanePAustin/pands-project2021/blob/main/Images/Iris_versicolor.jpg "Iris Versicolor")|![alt text](https://github.com/ShanePAustin/pands-project2021/blob/main/Images/Iris_virginica.jpg "Iris Virginica")|

### 2.1 Libraries


## 3 Data Set

The data was found and downloaded form the UC Irvine Machine Learning Repository [4], it contains 4 attributes and 150 instances, split into 3 seperate classes (Iris species) of 50 each. A sample of the raw data is shown below:

```
5.1,3.8,1.6,0.2,Iris-setosa
4.6,3.2,1.4,0.2,Iris-setosa
5.3,3.7,1.5,0.2,Iris-setosa
5.0,3.3,1.4,0.2,Iris-setosa
7.0,3.2,4.7,1.4,Iris-versicolor
6.4,3.2,4.5,1.5,Iris-versicolor
6.9,3.1,4.9,1.5,Iris-versicolor
5.5,2.3,4.0,1.3,Iris-versicolor
6.5,2.8,4.6,1.5,Iris-versicolor
```

The four variables that were measured for each flower species are shown below:

1. Sepal Length (cm)
2. Sepal Width (cm)
3. Petal Length (cm)
4. Petal Width (cm)


### 3.1 Data Summary

The data was imported into the program and the column names were defined:

``` python
filename = "./data/iris.csv"

df = pd.read_csv(filename, header = None, names = ["Sepal Length", "Sepal Width", "Petal Length", "Petal Width", "Species"])
```

#### 3.1.1 Data Integrity

With the data imported a summary was performed to assess the data integrity and discern the overall structure. To test data integrity three tests were performed to check for missing fields and data consistency. The tests performed were: 

1. Shape - This is a utility of NumPy to return the number of attributes and the length of each attribute.
2. Count - This expands on shape and shows the length of each attribute to check if no data is missing
3. Count per Species - This check the balance of the dataset to confirm each flower species has an equal length

The code and the ouput of those tests is shown here:

``` python
shape = df.shape

count = df.count()

countSpecies = df["Species"].value_counts()
```

```
Data Shape

(150, 5)

Data Count

Sepal Length    150
Sepal Width     150
Petal Length    150
Petal Width     150
Species         150
dtype: int64

Data Count per Species

Iris-versicolor    50
Iris-virginica     50
Iris-setosa        50
Name: Species, dtype: int64
```


The next test was to check the uniformity and order of the data, by using Python function "head()" the first defined rows can be output as a sample of the dataset. 

``` python
head = df.groupby("Species").head(5)
```

The image below shows the first five lines of each species:

```
Data Head per Species

     Sepal Length  Sepal Width  Petal Length  Petal Width          Species
0             5.1          3.5           1.4          0.2      Iris-setosa
1             4.9          3.0           1.4          0.2      Iris-setosa
2             4.7          3.2           1.3          0.2      Iris-setosa
3             4.6          3.1           1.5          0.2      Iris-setosa
4             5.0          3.6           1.4          0.2      Iris-setosa
50            7.0          3.2           4.7          1.4  Iris-versicolor
51            6.4          3.2           4.5          1.5  Iris-versicolor
52            6.9          3.1           4.9          1.5  Iris-versicolor
53            5.5          2.3           4.0          1.3  Iris-versicolor
54            6.5          2.8           4.6          1.5  Iris-versicolor
100           6.3          3.3           6.0          2.5   Iris-virginica
101           5.8          2.7           5.1          1.9   Iris-virginica
102           7.1          3.0           5.9          2.1   Iris-virginica
103           6.3          2.9           5.6          1.8   Iris-virginica
104           6.5          3.0           5.8          2.2   Iris-virginica
```

#### 3.1.2 Data Statistics

Once the integrity of the data has been established, a surface level statistical analysis can be performed using the Python function "describe()". Ths shows the count, mean, standard deviation, min, max and the quartile points of the sorted data. The code and result of this (rounded to 2 decimal places) is shown below:

```python
summary = df.describe().round(2)
```

```
Data Summary 

       Sepal Length  Sepal Width  Petal Length  Petal Width
count        150.00       150.00        150.00       150.00
mean           5.84         3.05          3.76         1.20
std            0.83         0.43          1.76         0.76
min            4.30         2.00          1.00         0.10
25%            5.10         2.80          1.60         0.30
50%            5.80         3.00          4.35         1.30
75%            6.40         3.30          5.10         1.80
max            7.90         4.40          6.90         2.50
```

This was then expanded to breakdown the statistical summary per species:

```python
sumSeto = df[df["Species"] == "Iris-setosa"].describe().round(2)
sumVers = df[df["Species"] == "Iris-versicolor"].describe().round(2)
sumVirg = df[df["Species"] == "Iris-virginica"].describe().round(2)
```

```
Data Summary (Setosa)

       Sepal Length  Sepal Width  Petal Length  Petal Width
count         50.00        50.00         50.00        50.00
mean           5.01         3.42          1.46         0.24
std            0.35         0.38          0.17         0.11
min            4.30         2.30          1.00         0.10
25%            4.80         3.12          1.40         0.20
50%            5.00         3.40          1.50         0.20
75%            5.20         3.68          1.58         0.30
max            5.80         4.40          1.90         0.60

Data Summary (Versicolor)

       Sepal Length  Sepal Width  Petal Length  Petal Width
count         50.00        50.00         50.00        50.00
mean           5.94         2.77          4.26         1.33
std            0.52         0.31          0.47         0.20
min            4.90         2.00          3.00         1.00
25%            5.60         2.52          4.00         1.20
50%            5.90         2.80          4.35         1.30
75%            6.30         3.00          4.60         1.50
max            7.00         3.40          5.10         1.80

Data Summary (Virginica)

       Sepal Length  Sepal Width  Petal Length  Petal Width
count         50.00        50.00         50.00        50.00
mean           6.59         2.97          5.55         2.03
std            0.64         0.32          0.55         0.27
min            4.90         2.20          4.50         1.40
25%            6.22         2.80          5.10         1.80
50%            6.50         3.00          5.55         2.00
75%            6.90         3.18          5.88         2.30
max            7.90         3.80          6.90         2.50
```

## 4 Data Analysis

### 4.1 Histograms

![alt text](https://github.com/ShanePAustin/pands-project2021/blob/main/Plots/SLHist.png "Sepal Length Histogram")

![alt text](https://github.com/ShanePAustin/pands-project2021/blob/main/Plots/SWHist.png "Sepal Width Histogram")

![alt text](https://github.com/ShanePAustin/pands-project2021/blob/main/Plots/PLHist.png "Petal Length Histogram")

![alt text](https://github.com/ShanePAustin/pands-project2021/blob/main/Plots/PWHist.png "Petal Width Histogram")

### 4.2 Scatterplots

![alt text](https://github.com/ShanePAustin/pands-project2021/blob/main/Plots/SLSWscatterPlot.png "SLSW Scatter Plot")

![alt text](https://github.com/ShanePAustin/pands-project2021/blob/main/Plots/PLPWscatterPlot.png "PLPW Scatter Plot")

![alt text](https://github.com/ShanePAustin/pands-project2021/blob/main/Plots/SLPLscatterPlot.png "SLPL Scatter Plot")

![alt text](https://github.com/ShanePAustin/pands-project2021/blob/main/Plots/SWPWscatterPlot.png "SWPW Scatter Plot")

### 4.3 Boxplots

### 4.4 Violinplots

## References

1) https://towardsdatascience.com/the-iris-dataset-a-little-bit-of-history-and-biology-fb4812f5a7b5

2) https://en.wikipedia.org/wiki/Iris_flower_data_set

3) https://digital.library.adelaide.edu.au/dspace/bitstream/2440/15227/1/138.pdf

4) https://archive.ics.uci.edu/ml/datasets/iris

5) https://rpubs.com/AjinkyaUC/Iris_DataSet

6) https://medium.com/@avulurivenkatasaireddy/exploratory-data-analysis-of-iris-data-set-using-python-823e54110d2d

7) https://www.researchgate.net/publication/344196553_Machine-learning_analysis_for_the_Iris_dataset

