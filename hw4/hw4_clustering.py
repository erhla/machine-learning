# -*- coding: utf-8 -*-
"""
Created on Mon May 27 19:52:44 2019

@author: erhla
"""

import pipeline
import pandas as pd
from sklearn.cluster import KMeans
from sklearn import tree

df = pipeline.read_load('/Users/erhla/Downloads/projects_2012_2013.csv')

#convert columns to datetime and add outcome column
df['date_posted'] = pd.to_datetime(df['date_posted'])
df['datefullyfunded'] = pd.to_datetime(df['datefullyfunded'])
df['days_to_fund'] = df['datefullyfunded'] - df['date_posted']
df['funded_within_60_days'] = pd.get_dummies(df['days_to_fund'] <= pd.Timedelta('60 days'), drop_first=True)

train_start_dates = ['06/01/2012', '12/01/2012', '06/01/2013']
date_col = 'date_posted'
test_length = '26 w'
test_train_offset = '60 d'
cols_to_fill = ['students_reached']
cols_to_drop_nas = ['primary_focus_area', 'resource_type', 'grade_level']
y_col = 'funded_within_60_days'

feature_dict = {'students_reached': 'discretized',
                'total_price_including_optional_support': 'discretized',
                'school_charter': 'dummy',
                'school_magnet': 'dummy',
                'eligible_double_your_impact_match': 'dummy',
                'teacher_prefix': 'dummy',
                'poverty_level': 'dummy',
                'grade_level': 'dummy',
                'primary_focus_area': 'dummy',
                'resource_type': 'dummy'
               }

train_start = train_start_dates[0]
test, train = pipeline.time_split(df, date_col, train_start, test_length, test_train_offset)
train = pipeline.preprocess(train, cols_to_fill, cols_to_drop_nas)
test = pipeline.preprocess(test, cols_to_fill, cols_to_drop_nas)
train, feature_ls = pipeline.generate_features(train, feature_dict, 10)
test, feature_ls2 = pipeline.generate_features(test, feature_dict, 10)
x_cols = list(set(feature_ls) & set(feature_ls2)) #include only feature columns which appear in both testing/training

kmean = KMeans().fit(train[x_cols])
train['pred_label'] = pd.Series(kmean.labels_, index=train.index)

def tst():
    for label, grouped in train.groupby('pred_label'):
        pipeline.explore(grouped, x_cols)
    for label, grouped in train.groupby('pred_label'):
        tmptree = tree.DecisionTreeClassifier(max_depth=4)
        tmptree.fit(grouped[x_cols], grouped[y_col])
        print('\n\n', label, '\n\n')
        for i in range(len(x_cols)):
            if tmptree.feature_importances_[i] > 0.2:
                print(x_cols[i], 'has ', tmptree.feature_importances_[i])

def merge(df, clusters_to_merge):
    unused_label = max(df.pred_label) + 1
    df[df['pred_label'].isin(clusters_to_merge)]['pred_label'] = unused_label
    return df

def recluster(df, k):
    kmean = KMeans(n_clusters=k).fit(df[x_cols])
    df['pred_label'] =  pd.Series(kmean.labels_, index=df.index)
    return df

def split_cluster(df, cluster_to_split, k):
    unused_label = max(df.pred_label) + 1
    cluster = df[df['pred_label'] == cluster_to_split]
    kmean = KMeans(n_clusters=k).fit(cluster[x_cols])
    df[df['pred_label'] == cluster_to_split]['pred_label'] = pd.Series(kmean.labels_, index=cluster.index) + unused_label

    return df
'''    
results = []
for train_start in train_start_dates:
    test, train = pipeline.time_split(df, date_col, train_start, test_length, test_train_offset)
    train = pipeline.preprocess(train, cols_to_fill, cols_to_drop_nas)
    test = pipeline.preprocess(test, cols_to_fill, cols_to_drop_nas)
    train, feature_ls = pipeline.generate_features(train, feature_dict, 10)
    test, feature_ls2 = pipeline.generate_features(test, feature_dict, 10)
    x_cols = list(set(feature_ls) & set(feature_ls2)) #include only feature columns which appear in both testing/training
    eval_metrics = pipeline.build_models(test[x_cols], test[y_col], train[x_cols], train[y_col])
    eval_metrics['train_start'] = train_start
    results.append(eval_metrics)
    
total = pd.concat(results)
total.to_excel('results.xlsx')
'''
