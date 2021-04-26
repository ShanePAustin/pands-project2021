##### pands-project2021

# Iris Flower Data Set

### Programming and Scripting 52167

### Shane Austin G00318488

## Table of Contents

1. [Problem Statement](#1-problem-statement)
2. [Introduction](#2-introduction)
    1. [Libraries](#2.1-libraries)
3. [Data Set](#3-data-set)
    1. [Data Summary](#31-data-summary)
        1. [Data Integrity](#311-data-integrity)
        2. [Data Statistics](#312-data-statistics)
4. [Data Analysis](#4-data-analysis)
    1. [Histograms](#41-histograms)
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

The images above taken from wikipedia [2] show the three subspecies of Iris 

### 2.1 Libraries

```python
import csv
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
```

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

The three species of Iris are:

* Iris Setosa
* Iris Versicolor
* Iris Virginica


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
The output confirms that the data consists of 150 rows and 5 columns. Each column contains 150 rows and each species contains 50 rows each. This is what was expected from the description of the dataset and confirms that there are no missing fields.

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

The table below shows the first five lines of each species, from this it can be seen that the data is ordered and in the expected order:

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
The data shows the largest deviation of data is present with the petal length variable, highligthed by the largest range between min and max lengths (1 -6.9) compared to sepal length being (4.3-7.9)

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

While the above is informative, it is more benefical to perform the same analyis seperate by species. Therefore the describe() function was expanded to breakdown the statistical summary per species. The code to do this is layed out below:

```python
sumSeto = df[df["Species"] == "Iris-setosa"].describe().round(2)
sumVers = df[df["Species"] == "Iris-versicolor"].describe().round(2)
sumVirg = df[df["Species"] == "Iris-virginica"].describe().round(2)
```

When the data is split like this it is easier to see the reason for the large range with petal length discussed above. The petal lengths on the Iris Setosa are significantly shorter than with the other 2 species. It is also evident that the standard deviations are much lower meaning there us less variation from the mean.

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

Correlation

Finally a correlation was performed on the data to assess the relationships between the variables. The closer to 1 the stronger the correlation exists between the two variables.

```
Data Correlation

              Sepal Length  Sepal Width  Petal Length  Petal Width
Sepal Length      1.000000    -0.109369      0.871754     0.817954
Sepal Width      -0.109369     1.000000     -0.420516    -0.356544
Petal Length      0.871754    -0.420516      1.000000     0.962757
Petal Width       0.817954    -0.356544      0.962757     1.000000
```

The code shown utilised Seaborn to create a heatmap of the table above, The darker the colour the stronger the correlation. This is the first example of the functionality of python to visualise data, which turns a table of numbers into an easily interpertable image.

```python
    sns.heatmap(df.corr(),cmap="BuPu", annot=True)
    plt.savefig("./plots/hMap.png")
```

A quick glance at the heatmap below shows a strong correlation between the Variable pairs:

* Petal length - Petal width
* Petal length - Sepal length
* Petal Width - Sepal length

![alt text](https://github.com/ShanePAustin/pands-project2021/blob/main/Plots/hMap.png "Heat Map")

This data was all saved to a seperate text file using the code below:

```python
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
```

## 4 Data Analysis

### 4.1 Histograms

The first plot to perform is to visualise the distribution of each variable, the following four plots are Histograms of each variable divided into ten bins. 

![alt text](https://github.com/ShanePAustin/pands-project2021/blob/main/Plots/SLHist.png "Sepal Length Histogram")

The highest frequencies of Sepal Length are around 5.5cm and 6.25cm both with a count over 25, there is another peak around 4.75-5cm of 23. These are likely to indicate the means of the three seperate species. 

![alt text](https://github.com/ShanePAustin/pands-project2021/blob/main/Plots/SWHist.png "Sepal Width Histogram")

Sepal Width has a more standard distribution with the highest frequency around 3cm with a count of over 35 instances.

![alt text](https://github.com/ShanePAustin/pands-project2021/blob/main/Plots/PLHist.png "Petal Length Histogram")

Petal Length has two distinct peaks the largest is around 1-1.5cm, this is likely due to the trend that Iris Setosa petals are significantly shorter than the other 2 subspecies. 

![alt text](https://github.com/ShanePAustin/pands-project2021/blob/main/Plots/PWHist.png "Petal Width Histogram")

Petal Width also has three distinct peaks which are likely attributed to the 3 subspecies.

The sample code for one Histogram is shown below:

```python
    ax = plt.axes()
    plt.hist(df["Sepal Length"], color = "#810f7c", ec = "black")
    plt.title("Sepal Length Histogram")
    plt.xlabel("Sepal Length")
    ax.set_facecolor("lightgrey")
```

![alt text](https://github.com/ShanePAustin/pands-project2021/blob/main/Plots/distPlot.png "Dist Plot")

```python
    distplot, axes = plt.subplots(2,2, figsize=(10,10), sharex=False)
    sns.histplot( df["Sepal Length"] , kde = True, stat="density", linewidth=1, color="#810f7c", ax=axes[0, 0])
    sns.histplot( df["Sepal Width" ] , kde = True, stat="density", linewidth=1, color="#8856a7" ,ax=axes[0, 1]) 
    sns.histplot( df["Petal Length"] , kde = True, stat="density", linewidth=1, color="#8c96c6", ax=axes[1, 0]) 
    sns.histplot( df["Petal Width" ] , kde = True, stat="density", linewidth=1, color="#b3cde3", ax=axes[1, 1])
    plt.suptitle("Combined Histograms and Distribution Estimation")
```


### 4.2 Scatterplots

![alt text](https://github.com/ShanePAustin/pands-project2021/blob/main/Plots/SLSWscatterPlot.png "SLSW Scatter Plot")

![alt text](https://github.com/ShanePAustin/pands-project2021/blob/main/Plots/PLPWscatterPlot.png "PLPW Scatter Plot")

![alt text](https://github.com/ShanePAustin/pands-project2021/blob/main/Plots/SLPLscatterPlot.png "SLPL Scatter Plot")

![alt text](https://github.com/ShanePAustin/pands-project2021/blob/main/Plots/SWPWscatterPlot.png "SWPW Scatter Plot")

```python
    sns.FacetGrid(df,hue="Species", palette="BuPu_r", height=5).map(plt.scatter, "Sepal Length", "Sepal Width" ).add_legend()
    plt.title("Sepal Length / Sepal Width Scatter Plot")
    plt.subplots_adjust(top=0.9)
```

![alt text](https://github.com/ShanePAustin/pands-project2021/blob/main/Plots/scatterPlot.png "Pair Plot")

```python
sns.pairplot(df,hue="Species",height=3)
```

### 4.3 Boxplots

![alt text](https://github.com/ShanePAustin/pands-project2021/blob/main/Plots/boxPlots.png "Box Plot")

```python
    boxPlot, axes = plt.subplots(1,4, figsize=(16,8))
    sns.boxplot(x="Species", y="Sepal Length", hue="Species", data=df, palette="BuPu_r", ax=axes[0], dodge=False)    
    sns.boxplot(x="Species", y="Sepal Width" , hue="Species", data=df, palette="BuPu_r", ax=axes[1], dodge=False)
    sns.boxplot(x="Species", y="Petal Length", hue="Species", data=df, palette="BuPu_r", ax=axes[2], dodge=False)
    sns.boxplot(x="Species", y="Petal Width" , hue="Species", data=df, palette="BuPu_r", ax=axes[3], dodge=False)
    plt.suptitle("Box Plots")
```


### 4.4 Violinplots

![alt text](https://github.com/ShanePAustin/pands-project2021/blob/main/Plots/violinPlots.png "Violin Plot")

```python
    violinplot, axes = plt.subplots(2,2, figsize=(10,10), sharex=False)
    sns.violinplot(x="Species", y="Sepal Length", hue="Species", data=df, palette="BuPu_r", ax=axes[0,0], dodge=False)
    sns.violinplot(x="Species", y="Sepal Width" , hue="Species", data=df, palette="BuPu_r", ax=axes[0,1], dodge=False)
    sns.violinplot(x="Species", y="Petal Length", hue="Species", data=df, palette="BuPu_r", ax=axes[1,0], dodge=False)
    sns.violinplot(x="Species", y="Petal Width" , hue="Species", data=df, palette="BuPu_r", ax=axes[1,1], dodge=False)
    plt.suptitle("Violin Plots")
```

## References

1) https://towardsdatascience.com/the-iris-dataset-a-little-bit-of-history-and-biology-fb4812f5a7b5

2) https://en.wikipedia.org/wiki/Iris_flower_data_set

3) https://digital.library.adelaide.edu.au/dspace/bitstream/2440/15227/1/138.pdf

4) https://archive.ics.uci.edu/ml/datasets/iris

5) https://rpubs.com/AjinkyaUC/Iris_DataSet

6) https://medium.com/@avulurivenkatasaireddy/exploratory-data-analysis-of-iris-data-set-using-python-823e54110d2d

7) https://www.researchgate.net/publication/344196553_Machine-learning_analysis_for_the_Iris_dataset

