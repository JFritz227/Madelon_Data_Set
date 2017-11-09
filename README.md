# Project 3: Feature Selection + Classification

### Domain and Data

You're working as a data scientist with a research firm. You're firm is bidding on a big project that will involve working with thousands or possibly tens of thousands of features. You know it will be impossible to use conventional feature selection techniques. You propose that a way to win the contract is to demonstrate a capacity to identify relevant features using machine learning. Your boss says, "Great idea. Write it up." You figure that working with a synthetic dataset such as [Madelon](https://archive.ics.uci.edu/ml/datasets/Madelon) is an excellent way to demonstrate your abilities. 

#### Requirement

This work must be done on AWS.

### Problem Statement

Your challenge here is to develop a series of models for two purposes:

1. for the purposes of identifying relevant features. 
2. for the purposes of generating predictions from the model. 

### Solution Statement

Your final product will consist of:

1. A prepared report
2. A series of Jupyter notebooks to be used to control your pipelines

### Tasks

#### Data Manipulation

You should do substantive work on at least six subsets of the data. 

- 3 sets of 10% of the data from the UCI Madelon set

    ```
    -rw-r--r-- 1 jovyan users 3601794 Oct 25 23:30 madelon_test.txt
    -rw-r--r-- 1 jovyan users    4999 Oct 26 16:30 madelon_train_targets.txt
    -rw-r--r-- 1 jovyan users 4001993 Oct 26 16:48 madelon_train.txt
    -rw-r--r-- 1 jovyan users    1499 Oct 26 16:29 madelon_valid_targets.txt
    -rw-r--r-- 1 jovyan users 1200598 Oct 25 23:31 madelon_valid.txt
    ```
    
    train = 2000 observations
    validation = 600 observations
    test = 1800 observations
    
    10% train sample = 200 observations
    10% validation sample = 60 observations
    10% test sample = 180 observations
    
- 3 sets of 10% of the data from the Madelon set made available by your instructors

    90% confidence interval with 1% margin of error for 220,000 observations requires a sample size of 6,525 observations.
    95% confidence interval with 1% margin of error for 220,000 observations requires a sample size of 9,203 observations.
    99% confidence interval with 1% margin of error for 220,000 observations requires a sample size of 15,471 observations.
    
##### Prepared Report

Your report should:

1. be a pdf
2. include EDA of each subset 
   - EDA needs may be different depending upon subset or your approach to a solution
3. present results from Step 1: Benchmarking

    Benchmarked using the average train and test scores from 3 different 10% samples of the small dataset (2000 train, 600 
    validation). The data entered was both raw and scaled data and the models used were KNeighborsRegressor/Classifier, 
    DecisionTreeClassifier, RandomForestClassifier, and LogisticRegression.
    
4. present results from Step 2: Identify Salient Features
5. present results from Step 3: Feature Importances
6. present results from Step 4: Build Model

##### Jupyter Notebook, EDA 

- perform EDA on each set as you see necessary

##### Jupyter Notebook, Step 1 - Benchmarking
- build pipeline to perform a naive fit for each of the base model classes:
	- logistic regression
	- decision tree
	- k nearest neighbors
	- support vector classifier
- in order to do this, you will need to set a high `C` value in order to perform minimal regularization, in the case of logistic regression and support vector classifier.

##### Jupyter Notebook, Step 2 - Identify Features
- Build feature selection pipelines using at least three different techniques
- **NOTE**: these pipelines are being used for feature selection not prediction

##### Jupyter Notebook, Step 3 - Testing Model Pipelines
- Considering these results, develop a strategy for building a final predictive model
- recommended approaches:
    - Use feature selection to reduce the dataset to a manageable size then use conventional methods
    - Use dimension reduction to reduce the dataset to a manageable size then use conventional methods
    - Use an iterative model training method to use the entire dataset
- This notebook should be a "playground" where you try various approaches to solving this problem
   
##### Jupyter Notebook, Step 4 - Build Model
- Implement your final model
- (Optionally) use the entire data set

---

### Requirements

- Many Jupyter Notebooks
- A written report of your findings that detail the accuracy and assumptions of your model.

---

### Suggestions

- Document **everything**.