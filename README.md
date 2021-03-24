# GMM-and-K-Means_Iris
Gaussian Mixture Model and K-Means algorithm to predict flower classes from the famous iris dataset.

## Table of Contens
* [Description](#description)
* [Dataset](#dataset)
* [Setup](#setup)
* [Run code](#run-code)

## Description
K-Means and Gaussian Mixture Model clustering algorithm to assign iris flowers to 3 different classes depending on 4 attributes.
I'm still working on using the K-Means clustering to initialize the dataset to increase my GMM accuracy which is now stuck at 82%.
The K-Means clustering algorithm achieved an accuracy of 84% and the (future) GMM ~93%.

## Technologies
Project is created with:
* Python version: 3.9.1
* NumPy library version : 1.20.0
* Pandas library version : 1.2.2

## Dataset
The dataset is quite simple. It contains a list of 150 points with four attributes per point.
There are three different classes:
* Iris Setosa
* Iris Versicolour
* Iris Virginica

and four different attributes that our model uses to cluster the flowers:
* sepal length in cm
* sepal width in cm
* petal length in cm
* petal width in cm

## Setup
Download the .py files, datasets, and different cluster prediction files.
Place all files in single folder or project folder.

## Run Code

I have added the following code outside both classes to run the models. It also writes the label assignements to the corresponding file.
```
x = GMM('Data.tsv', 3)
labels = x.GMM()
with open('GMM_output.tsv', 'w', newline='') as f_output:
    tsv_output = csv.writer(f_output, delimiter='\t')
    tsv_output.writerow(labels)
 ```
Enjoy!
