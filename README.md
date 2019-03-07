# K-NN and Naive Bayes classifiers for text document analysis using Python

This repository consists of implementation of two types of classifiers with testing module and visualization.

## Prerequisites

This project uses numpy for computation, pickle for data managment and matplotlib for visualization of results.

List of all packages used:
* numpy
* logging
* pickle
* sys
* time
* matplotlib
* warnings
* scipy.spatial

## Content.py

File consisting of needed methods.

For k-NN classifier:
* Hamming distance - with two implementation (faster using scipy and much slower using only numpy)
* train label sorter for k-NN - using results from Hamming distance
* p(y|x) for knn - calculating conditional probability
* classification error
* model selection k-NN

For Naive Bayes classifier:
* estimate a priori
* estimate p(x|y) for Naive Bayes
* estimate p(y|x) for Naive Bayes
* model selection Naive Bayes

Inputs and outputs of all methods are described in the comments in the code.

## Test.py

File consisting of unittests for methods in content.py. Expected result for all tests is "ok".

## Main.py

From main the tests are run and model training is performed. 
Naive Bayes model is tested with following parameters a and b:

```
a_values = [1, 3, 10, 30, 100, 300, 1000]
b_values = [1, 3, 10, 30, 100, 300, 1000]
```
k-NN model is trained with following number of neighbours:

```
k_values = range(1, 201, 2)
```
Results are presented on graphs prepared with matplotlib.
Comparison of methods is performed.

## Acknowledgments

This project was prepared as a part of Machine Learning course in Wrocław University of Technology.


