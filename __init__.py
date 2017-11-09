import warnings
warnings.filterwarnings('ignore')

import psycopg2 as pg2
from psycopg2.extras import RealDictCursor

from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.feature_selection import SelectKBest, SelectFromModel, RFE, SelectPercentile, f_classif
from statsmodels.formula.api import ols
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.svm import LinearSVC
from sklearn import metrics
from sklearn.decomposition import PCA

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats as st
from scipy.stats import boxcox


madelon_train = pd.read_csv('data/madelon_train.txt', sep = ' ', header = None)
madelon_valid = pd.read_csv('data/madelon_valid.txt', sep = ' ', header = None)
madelon_test = pd.read_csv('data/madelon_test.txt', sep = ' ', header = None)

madelon_train_targets = pd.read_csv('data/madelon_train_targets.txt', sep = ' ', header = None)
madelon_val_target = pd.read_csv('data/madelon_valid_targets.txt', sep = ' ', header = None)

del madelon_train[500]
del madelon_valid[500]
del madelon_test[500]

def add_train_test_score_df(train_df, val_df, model, scaled=False, feats = 20):

    i = 42

    train_scores = list()
    test_scores = list()
 
    if scaled:
        ss_train = StandardScaler()
        train_df_sc = ss_train.fit_transform(train_df)
        
        ss_val = StandardScaler()
        val_df_sc = ss_val.fit_transform(val_df)
        
        train_df = pd.DataFrame(train_df_sc)
        val_df = pd.DataFrame(val_df_sc)
        
        train_df = train_df.rename(columns = {feats:'targets'})
        val_df = val_df.rename(columns = {feats:'targets'})
        
    for samples in range(3):
        train_sample = train_df.sample(frac = 0.1, random_state = i+samples)
        val_sample = val_df.sample(frac = 0.1, random_state = i+samples)
    
        train_targets = train_sample['targets']
        train_sample = train_sample.drop('targets', 1)
    
        val_targets = val_sample['targets']
        val_sample = val_sample.drop('targets', 1)
    
        mod = model
        mod.fit(train_sample, train_targets)

        train_scores.append(mod.score(train_sample, train_targets))
        test_scores.append(mod.score(val_sample, val_targets))

    avg_train_score = sum(train_scores)/len(train_scores)
    avg_test_score = sum(test_scores)/len(test_scores)

    return [avg_train_score, avg_test_score]



# function that drops one feature and tries to predict that feature using just the other 499 features. Uses two different models to do this and returns the test score (R2) value of each model for the selected feature

def calculate_r_2_for_feature(data, feature):
    new_data = data.drop(feature, axis=1)

    X_train, \
    X_test,  \
    y_train, \
    y_test = train_test_split(
        new_data, data[feature] ,test_size=0.25,
        random_state = 42
    )

    decision_tree = DecisionTreeRegressor(random_state = 42)
    decision_tree.fit(X_train, y_train)
    
    kneighbors = KNeighborsRegressor()
    kneighbors.fit(X_train, y_train)

    decision_tree_score = decision_tree.score(X_test, y_test)
    kneighbors_score = kneighbors.score(X_test, y_test)
    
    return decision_tree_score, kneighbors_score




# creates a dataframe of R2 values from a set of data and every column in that set, being passed through the R2 function created above

def create_r2_dataframe(data):
    r2_list = list()
    
    for cols in data:
        r2_list.append(calculate_r_2_for_feature(data, cols))
    
    r2_array = np.array(r2_list)
    
    r2_df = pd.DataFrame(r2_array)
    
    return r2_df