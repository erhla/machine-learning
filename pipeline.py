'''
pipeline.py is a machine learning pipeline
Eric Langowski
version 0.0
'''


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.model_selection import train_test_split, ParameterGrid
from sklearn.metrics import accuracy_score
from sklearn import tree, svm
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, BaggingClassifier, AdaBoostClassifier

#Add boosting and bagging

#read/load data
def read_load(file_path):
    '''
    Currently only takes csvs

    Input: str, filepath
    Output: dataframe
    '''
    return pd.read_csv(file_path)

#explore data
def explore(df, exclude=''):
    '''
    Gives some statistics on the data; note that descriptive columns
    should be passed as a list to exclude to prevent nonsensical analysis

    Input: df: dataframe
           exclude: list of column names
    '''
    if exclude: #drop descriptive columns
        df = df.drop(exclude, axis=1)
    for col_name in df.columns:
        description = df[col_name].describe()
        print('VARIABLE:', col_name, '\n', description, '\n')

    corrs = df.corr() #find correlations
    for col_name in corrs.index:
        series = corrs[col_name]
        filtered = series[(abs(series) >= 0.25) & (abs(series) != 1.0)]
        if not filtered.empty:
            print('correlation with:', col_name, '\n\n', filtered, '\n\n')
    plt.matshow(corrs)
    plt.xticks(range(len(corrs.columns)), corrs.columns)
    plt.yticks(range(len(corrs.columns)), corrs.columns)
    plt.show() #Correlation Graphs
    for col_name in df.columns: #check for outliers
        current = df[col_name].dropna()
        outliers = current[(np.abs(stats.zscore(current)) > 5)] #find values 5+ sds
        if not outliers.empty:
            sns.distplot(df[col_name], hist=False, rug=True)
            print(col_name, 'has possible outliers', outliers.shape[0], '\n')

#pre-process data
def preprocess(df, skipcols=''):
    '''
    replace missing values with the column's median value. descriptive columns
    can be passed as a list to skipcols

    Input: df: dataframe
           skipcols: list of column names

    Output: dataframe
    '''
    for col_name in df.columns:
        if col_name not in skipcols:
            count = df[df[col_name].isna()].shape[0]
            fill_val = df[col_name].agg('median')
            df[col_name] = df[col_name].fillna(fill_val)
            if count > 0:
                print(count, 'nas filled for', col_name)
    return df

#generate features
def generate_features(df, col_name, feature_type, division_num=4):
    '''
    generate features updates a columns in the dataframe as a features
    either a discretized (from a continuous variable) or dummy (from a
    categorical variable)

    Input: df: dataframe
           col_name: str, column name
           feature_type: str, either 'discretize' or 'dummy'
           division_num: int, optional parameter representing the number
                         of bins to use to discretize

    Returns: dataframe
    '''
    if feature_type == 'discretized':
        bins = [] #quantile bins
        i = 0
        while i <= 1:
            bins.append(i)
            i = i + 1 / division_num
        labels = list(range(len(bins)))[1:] #bin labels
        df[col_name] = pd.qcut(df[col_name], bins, labels=labels)
        df[col_name] = df[col_name].fillna(labels[-1])
        print(col_name, 'discretized')
    if feature_type == 'dummy':
        unique = df[col_name].unique()
        print(col_name, 'has values: ', unique)
        df[col_name] = df[col_name].astype('category')
        if len(unique) != 2:
            dummies = pd.get_dummies(df[col_name], prefix=col_name)
            df[dummies.columns] = dummies
            print('target variable has more than two values, multiple dummies created')
        else:
            df[col_name] = df[col_name].cat.rename_categories([-1, 1])
            print('dummy created for', col_name)
    return df


def define_models(df, X_cols, y_col, test_size):
    models = {'RF': RandomForestClassifier(),
              'LR': LogisticRegression(),
              'SVM': svm.SVC(),
              'DT': tree.DecisionTreeClassifier(),
              'KNN': KNeighborsClassifier(),
              'GB': GradientBoostingClassifier(),
              'BC': BaggingClassifier(tree.DecisionTreeClassifier()),
              'ABC': AdaBoostClassifier(tree.DecisionTreeClassifier())
              }
    parameters = {
            'RF': {'n_estimators':[10, 100], 'max_depth':[5,20], 'min_samples_split':[10]},
            'LR': {'penalty':['l1', 'l2'], 'C':[0.001, 0.1, 1]},
            'SVM': {'C': [0.001, 0.1, 1], 'kernel': ['linear']},
            'DT': {'criterion': ['gini', 'entropy'], 'max_depth': [5, 10, 20], 'min_samples_split': [10]},
            'KNN': {'n_neighbors': [1, 10, 100], 'weights': ['uniform', 'distance'], 'algorithm': ['auto']},
            'GB': {'n_estimators': [10, 100], 'learning_rate' : [0.001, 0.1], 'subsample' : [0.1, 1.0], 'max_depth': [5, 50]},
            'BC': {'n_estimators': [10, 100, 10000]},
            'ABC': {'algorithm': ['SAMME'], 'n_estimators': [10, 100, 10000]}
            }
    X = df[X_cols]
    Y = df[y_col]
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=test_size)

    for model_type in models:
        for p in ParameterGrid(parameters[model_type]):
            new_model = models[model_type]
            new_model.set_params(**p)
            
            evaluate_classifier(new_model, x_train, x_test, y_train, y_test)
            
        
        
    pass
    



#build classifier
def build_classifier(df, outcome_col, test_size, max_depth, min_split_size):
    '''
    build_classifier (for decision trees only) has multiple parameters and
    returns a complete model and testing data

    Input:
        df: a dataframe
        outcome_col: (str) name of outcome column
        test_size: (float) a fraction of data to be withheld to test
        max_depth: (int) maximal number of tree divisions allowed
        min_split_size: (int) minimum leaf size allowed to split

    Returns:
        dt_tree: (Tree) a modeled decision tree on training data
        x_test: (df) feature testing data
        y_test: (df) outcome testing data

    inspired by http://dataaspirant.com/2017/02/01/decision-tree-algorithm-python-with-scikit-learn/
    '''
    col_set = set(df.columns)
    col_set.remove(outcome_col)
    X, Y = df[list(col_set)], df[outcome_col]
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=test_size)
    dt_tree = tree.DecisionTreeClassifier(max_depth=max_depth, min_samples_split=min_split_size)
    dt_tree.fit(x_train, y_train)
    return dt_tree, x_test, y_test

#evaluate classifier
def evaluate_classifier(model, x_train, x_test, y_train, y_test):
    '''
    evaluate_classifier takes a fitted tree and testing data
    and evaluates based on accuracy

    Input:
        fitted_tree: (DecisionTreeClassifier)
        x_test: (df) feature testing data
        y_test: (df) outcome testing data
    '''

    
    for clf in [model, bagged, boosted]:
        clf.fit(x_train, y_train)
        
    
    predicted = fitted_tree.predict(x_test)
    score = accuracy_score(y_test, predicted)
    print('model has accuracy', score)
