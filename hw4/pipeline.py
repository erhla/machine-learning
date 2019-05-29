'''
pipeline.py is a machine learning pipeline
Eric Langowski
version 1.0
'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import recall_score, precision_score, roc_auc_score, precision_recall_curve
from sklearn import tree, svm
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, \
                                BaggingClassifier, AdaBoostClassifier

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


#temporal factors
def time_split(df, date_col, test_start, test_length, test_train_offset, train_start=None):
    '''
    simple function to split a dataframe into training and testing data
    based on dates and date_col

    Note: train_end != test_start to account for outcome to happen

    df: a dataframe
    date_col: str of date column to split on
    test_start: str testing start date
    test_length: str length of test data (e.g. '26 w')
    test_train_offset: str offset b/t train_end and test_start (e.g. '60 d')
    train_start: default is beginning of df, otherwise str (e.g. '01/01/2012')

    will return two pairs of dataframes:
        -testing data is over time period (test_start, test_end)
        -training data is over time period (train_start, train_end)
    '''
    test_start = pd.Timestamp(test_start)
    test_end = test_start + pd.Timedelta(test_length)

    if not train_start: #if no start date for train given, set to beginning of df
        train_start = min(df[date_col])
    else:
        train_start = pd.Timestamp(train_start)

    train_end = test_start - pd.Timedelta(test_train_offset) #outcome offset

    if train_end == test_start:
        print('warning: train_end equals test_start')

    test = df[(df[date_col] >= test_start) & (df[date_col] < test_end)]
    train = df[(df[date_col] >= train_start) & (df[date_col] < train_end)]

    return test, train


#pre-process data
def preprocess(df, cols_to_fill, cols_to_drop_nas):
    '''
    offers some simple imputation methods; use after test-train split

    replace missing values with the column's median value or drop rows with missing values

    Input: df: dataframe
           cols_to_fill: ls of cols to fill
           cols_to_drop_nas: ls of cols to drop if nas

    Output: dataframe
    '''
    for col_name in cols_to_drop_nas: #remove nas
        df = df[df[col_name].notna()]
    for col_name in cols_to_fill: #fill nas with median
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
           feature_dict: a dictionary of the form col_name: feature_type
           division_num: parameter for number of discretized values to make

    Returns: dataframe, list of features
    '''
    feature_ls = [] #store feature column names for later

    for col_name, feature_type in feature_dict.items():
        if feature_type == 'discretized':
            bins = [] #make division_num discrete bins
            i = 0
            while i <= 1:
                bins.append(i)
                i = i + 1 / division_num
            labels = list(range(len(bins)))[1:]

            df[col_name] = pd.qcut(df[col_name], bins, labels=labels) #fill bins
            df[col_name] = df[col_name].fillna(labels[-1]) #fill values not binned
            print(col_name, 'discretized')
            feature_ls.append(col_name)

        if feature_type == 'dummy':
            unique = df[col_name].unique()
            print(col_name, 'has values: ', unique)
            df[col_name] = df[col_name].astype('category')
            if len(unique) != 2: #create multiple dummy columns
                dummies = pd.get_dummies(df[col_name], prefix=col_name)
                df[list(dummies.columns)] = dummies
                df = df.drop(col_name, axis=1)
                feature_ls = feature_ls + list(dummies.columns)
                print('target variable has more than two values, multiple dummies created')
            else: #binary dummy column
                df[col_name] = df[col_name].cat.rename_categories([-1, 1])
                print('dummy created for', col_name)
                feature_ls.append(col_name)

    return df, feature_ls

#build classifiers
def build_models(x_test, y_test, x_train, y_train, models_to_run='all'):
    '''
    build_models takes a dataframe and specific columns and returns
    evaluated models

    inputs: df: dataframe
            X_cols: list of features
            y_col: str of target column
            date_col: the column to split on
            test_start_dates: a list of dates to split on
            models_to_run: default all models, otherwise pass a list of model ids
    returns:
            df: dataframe with evaluated model characteristics
    '''
    models = {'RF': RandomForestClassifier(),
              'LR': LogisticRegression(solver='lbfgs'),
              'SVM': svm.LinearSVC(),
              'DT': tree.DecisionTreeClassifier(),
              'KNN': KNeighborsClassifier(),
              'GB': GradientBoostingClassifier(),
              'BC': BaggingClassifier(tree.DecisionTreeClassifier()),
              'ABC': AdaBoostClassifier(tree.DecisionTreeClassifier())
              }
    parameters = {
        'RF': {'n_estimators':[10, 100], 'max_depth':[5, 20], 'min_samples_split':[10]},
        'LR': {'penalty':['l2'], 'C':[0.5, 1], 'max_iter': [5000]},
        'SVM': {'C': [0.1, 1], 'loss':['hinge']},
        'DT': {'criterion': ['gini', 'entropy'], 'max_depth': [5, 10, 20], 'min_samples_split': [10]},
        'KNN': {'n_neighbors': [1, 10, 100], 'weights': ['uniform', 'distance'], 'algorithm': ['auto']},
        'GB': {'n_estimators': [10, 100], 'learning_rate' : [0.001, 0.1], 'max_depth': [5]},
        'BC': {'n_estimators': [10, 100]},
        'ABC': {'algorithm': ['SAMME'], 'n_estimators': [10, 100]}
        }
    results = []

    if models_to_run == 'all': #run all models
        selected_models = models
    else:
        selected_models = {}
        for mdl_type in models_to_run: #run selected models only
            selected_models[mdl_type] = models[mdl_type]

    for model_type in selected_models:
        for p in ParameterGrid(parameters[model_type]):
            current = {}
            current['type'] = model_type
            current['parameters'] = p
            new_model = clone(models[model_type])
            new_model.set_params(**p)
            new_model.fit(x_train, y_train)
            if model_type == 'SVM':
                y_test_predicted = new_model.decision_function(x_test)
            else:
                y_test_predicted = new_model.predict_proba(x_test)[:, 1]
            evals = evaluate_classifier(y_test, y_test_predicted)
            for metric in evals:
                current[metric] = evals[metric]
            results.append(current)
    return pd.DataFrame.from_records(results)

#evaluate classifier

#following four functions
#from https://github.com/rayidghani/magicloops/blob/master/mlfunctions.py
def joint_sort_descending(l1, l2):
    idx = np.argsort(l1)[::-1]
    return l1[idx], l2[idx]

def generate_binary_at_k(y_scores, k):
    cutoff_index = int(len(y_scores) * (k / 100.0))
    predictions_binary = [1 if x < cutoff_index else 0 for x in range(len(y_scores))]
    return predictions_binary

def precision_at_k(y_true, y_scores, k):
    y_scores_sorted, y_true_sorted = joint_sort_descending(np.array(y_scores), np.array(y_true))
    preds_at_k = generate_binary_at_k(y_scores_sorted, k)
    precision = precision_score(y_true_sorted, preds_at_k)
    return precision

def recall_at_k(y_true, y_scores, k):
    y_scores_sorted, y_true_sorted = joint_sort_descending(np.array(y_scores), np.array(y_true))
    preds_at_k = generate_binary_at_k(y_scores_sorted, k)
    recall = recall_score(y_true_sorted, preds_at_k)
    return recall

def evaluate_classifier(y_test, y_test_predicted):
    '''
        precision at different levels, recall at different levels,
        area under curve, and precision-recall curves).
    '''
    results = {}
    results['baseline'] = y_test.mean()
    k_lst = [0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5]
    for k in k_lst:
        results[str(k) + '_precision'] = precision_at_k(y_test, y_test_predicted, k)
        results[str(k) + '_recall'] = recall_at_k(y_test, y_test_predicted, k)
    results['auc_roc'] = roc_auc_score(y_test, y_test_predicted)
    results['pr_curve'] = precision_recall_curve(y_test, y_test_predicted)
    return results

def create_clusters(df, x_cols, k):
    '''
    creates clusters for given features and k total clusters
    returns original dataframe with a new column 'pred_label'
    that stores the clusters
    '''
    kmean = KMeans(n_clusters=k).fit(df[x_cols])
    df['pred_label'] = pd.Series(kmean.labels_, index=df.index)
    return df

def explore_clusters(df, x_cols, y_col):
    '''
    prints many statistics about clusters and uses a decision tree to
    determine what features are most important for a given cluster above
    a threshold importance_threshold
    '''
    for label, grouped in df.groupby('pred_label'):
        explore(grouped, x_cols)
    
def explore_clusters_2(df, x_cols, y_col, importance_threshold):
    '''
    prints many statistics about clusters and uses a decision tree to
    determine what features are most important for a given cluster above
    a threshold importance_threshold
    '''
    for label, grouped in df.groupby('pred_label'):
        tmptree = tree.DecisionTreeClassifier(max_depth=4)
        tmptree.fit(grouped[x_cols], grouped[y_col])
        print('\n\n', label, '\n\n')
        for i, col in enumerate(x_cols):
            if tmptree.feature_importances_[i] > importance_threshold:
                print(col, 'has feature importance: ', tmptree.feature_importances_[i])

def merge_cluster(df, clusters_to_merge):
    '''
    merges all clusters in list clusters_to_merge into a new cluster
    '''
    unused_label = max(df.pred_label) + 1
    df.loc[df['pred_label'].isin(clusters_to_merge), 'pred_label'] = unused_label
    return df

def split_cluster(df, x_cols, cluster_to_split, k):
    '''
    splits clusters from cluster_to_split into k clusters
    '''
    unused_label = max(df.pred_label) + 1
    cluster = df[df['pred_label'] == cluster_to_split]
    kmean = KMeans(n_clusters=k).fit(cluster[x_cols])
    df.loc[df['pred_label'] == cluster_to_split, 'pred_label'] = \
            pd.Series(kmean.labels_, index=cluster.index) + unused_label
    return df
