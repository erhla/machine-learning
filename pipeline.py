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
from sklearn.metrics import accuracy_score, f1_score, precision_recall_curve
from sklearn import tree, svm
from sklearn.base import clone
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
def explore(input_df, cols_to_include):
    '''
    Gives some statistics on the data for discrete variables only

    Input: df: dataframe
           cols_to_include: list
    '''
    input_df = input_df[cols_to_include]
    for col_name in input_df.columns:
        description = input_df[col_name].describe()
        print('VARIABLE:', col_name, '\n', description, '\n')

    corrs = input_df.corr() #find correlations
    for col_name in corrs.index:
        series = corrs[col_name]
        filtered = series[(abs(series) >= 0.25) & (abs(series) != 1.0)]
        if not filtered.empty:
            print('correlation with:', col_name, '\n\n', filtered, '\n\n')
    plt.matshow(corrs)
    plt.xticks(range(len(corrs.columns)), corrs.columns)
    plt.yticks(range(len(corrs.columns)), corrs.columns)
    plt.show() #Correlation Graphs
    for col_name in input_df.columns: #check for outliers
        current = input_df[col_name].dropna()
        outliers = current[(np.abs(stats.zscore(current)) > 5)] #find values 5+ sds
        if not outliers.empty:
            sns.distplot(input_df[col_name], hist=False, rug=True)
            print(col_name, 'has possible outliers', outliers.shape[0], '\n')

#pre-process data
def preprocess(df, cols_to_fill, cols_to_drop_nas):
    '''
    replace missing values with the column's median value or drop missing values

    Input: df: dataframe

    Output: dataframe
    '''
    for col_name in cols_to_drop_nas:
        df = df[df[col_name].notna()]
    for col_name in cols_to_fill:
        count = df[df[col_name].isna()].shape[0]
        fill_val = df[col_name].agg('median')
        df[col_name] = df[col_name].fillna(fill_val)
        if count > 0:
            print(count, 'nas filled for', col_name)

    return df


#generate features
def generate_features(df, feature_dict, division_num):
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
    feature_ls = []
    for col_name, feature_type in feature_dict.items():
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
            feature_ls.append(col_name)
        if feature_type == 'dummy':
            unique = df[col_name].unique()
            print(col_name, 'has values: ', unique)
            df[col_name] = df[col_name].astype('category')
            if len(unique) != 2:
                dummies = pd.get_dummies(df[col_name], prefix=col_name)
                df[list(dummies.columns)] = dummies
                df = df.drop(col_name, axis=1)
                feature_ls = feature_ls + list(dummies.columns)
                print('target variable has more than two values, multiple dummies created')
            else:
                df[col_name] = df[col_name].cat.rename_categories([-1, 1])
                print('dummy created for', col_name)
                feature_ls.append(col_name)
            
    return df, feature_ls

#build classifiers
def build_models(df, X_cols, y_col, models_to_run='all'):
    models = {'RF': RandomForestClassifier(),
              'LR': LogisticRegression(),
              'SVM': svm.LinearSVC(),
              'DT': tree.DecisionTreeClassifier(),
              'KNN': KNeighborsClassifier(),
              'GB': GradientBoostingClassifier(),
              'BC': BaggingClassifier(tree.DecisionTreeClassifier()),
              'ABC': AdaBoostClassifier(tree.DecisionTreeClassifier())
              }
    parameters = {
            'RF': {'n_estimators':[10, 100], 'max_depth':[5,20], 'min_samples_split':[10]},
            'LR': {'penalty':['l1', 'l2'], 'C':[0.001, 0.1, 1]},
            'SVM': {'C': [0.01, 0.1, 1], 'loss':['hinge', 'squared_hinge']},
            'DT': {'criterion': ['gini', 'entropy'], 'max_depth': [5, 10, 20], 'min_samples_split': [10]},
            'KNN': {'n_neighbors': [1, 10, 100], 'weights': ['uniform', 'distance'], 'algorithm': ['auto']},
            'GB': {'n_estimators': [10, 100], 'learning_rate' : [0.001, 0.1], 'subsample' : [0.1, 1.0], 'max_depth': [5, 50]},
            'BC': {'n_estimators': [10, 100, 10000]},
            'ABC': {'algorithm': ['SAMME'], 'n_estimators': [10, 100, 10000]}
            }
    results = []
    
    for split_date in ['06/01/2012', '12/01/2012', '06/01/2013']:
        x_train, x_test, y_train, y_test = time_split(df, X_cols, y_col, split_date)
        
        if models_to_run == 'all': #run all models
            selected_models = models
        else:
            selected_models = {}
            for type in models_to_run:
                selected_models[type] = models[type]
        
        for model_type in selected_models:
            for p in ParameterGrid(parameters[model_type]):
                current = {}
                current['type'] = model_type
                current['parameters'] = p
                current['split_date'] = split_date
                new_model = clone(models[model_type])
                new_model.set_params(**p)
                new_model.fit(x_train, y_train)
                y_test_predicted = new_model.predict(x_test)
                evals = evaluate_classifier(new_model, y_test, y_test_predicted)
                for metric in evals:
                    current[metric] = evals[metric]
                results.append(current)
     
    return results

#temporal factors
    
def time_split(df, x_cols, y_col, split_date):
    date = pd.Timestamp(split_date)
    test = df[(df['date_posted'] > date) & (df['date_posted'] < date + pd.Timedelta('26 w'))]
    train = df[df['date_posted'] < date]
    
    x_test = test[x_cols]
    y_test = test[y_col]
    x_train = train[x_cols]
    y_train = train[y_col]
    
    return x_train, x_test, y_train, y_test
        
        

    
#evaluate classifier
def evaluate_classifier(model, y_test, y_test_predicted):
    '''
        precision at different levels, recall at different levels, 
        area under curve, and precision-recall curves).
    '''
    accuracy = accuracy_score(y_test, y_test_predicted)
    f1 = f1_score(y_test, y_test_predicted)
    pr_curve = precision_recall_curve(y_test, y_test_predicted)
    
    return {'accuracy': accuracy, 'f1': f1, 'pr_curve': pr_curve}
    
