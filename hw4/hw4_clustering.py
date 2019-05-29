# -*- coding: utf-8 -*-
import pipeline
import pandas as pd

df = pipeline.read_load('/Users/erhla/Downloads/projects_2012_2013.csv')
#convert columns to datetime and add outcome column
df['date_posted'] = pd.to_datetime(df['date_posted'])
df['datefullyfunded'] = pd.to_datetime(df['datefullyfunded'])
df['days_to_fund'] = df['datefullyfunded'] - df['date_posted']
df['funded_within_60_days'] = pd.get_dummies(df['days_to_fund'] <= pd.Timedelta('60 days'), drop_first=True)

#constants, hardcoded factors, and pre-selected features
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
#creating master df
df = pipeline.preprocess(df, cols_to_fill, cols_to_drop_nas)
df, feature_ls = pipeline.generate_features(df, feature_dict, 10)

#creating clusters
clustered = df.copy()
clustered = pipeline.create_clusters(clustered, feature_ls, 5)

pipeline.explore_clusters_2(clustered, feature_ls, y_col, 0.2)

#example merging clusters 2 and 3
tmp = pipeline.merge_cluster(clustered, [2, 3])
tmp['pred_label'].unique()

#example splitting cluster 4 into three clusters (renumbers as clusters 6 through 8)
tmp2 = pipeline.split_cluster(clustered, feature_ls, 4, 3)
tmp2['pred_label'].unique()

#example recluster
tmp3 = pipeline.create_clusters(clustered, feature_ls, 10)
tmp3['pred_label'].unique()

#import results from hw5
results = pd.read_excel('../hw5/results.xlsx')

#model with top 5% precision
top_model = results.loc[results['5_precision'].idxmax(axis=1)]

#getting testing and training data top model was trained/tested on
new_df = pipeline.read_load('/Users/erhla/Downloads/projects_2012_2013.csv')

#convert columns to datetime and add outcome column
new_df['date_posted'] = pd.to_datetime(new_df['date_posted'])
new_df['datefullyfunded'] = pd.to_datetime(new_df['datefullyfunded'])
new_df['days_to_fund'] = new_df['datefullyfunded'] - new_df['date_posted']
new_df['funded_within_60_days'] = pd.get_dummies(new_df['days_to_fund'] <= pd.Timedelta('60 days'), drop_first=True)
test, train = pipeline.time_split(new_df, date_col, top_model['train_start'], test_length, test_train_offset)
train = pipeline.preprocess(train, cols_to_fill, cols_to_drop_nas)
test = pipeline.preprocess(test, cols_to_fill, cols_to_drop_nas)
train, feature_ls = pipeline.generate_features(train, feature_dict, 10)
test, feature_ls2 = pipeline.generate_features(test, feature_dict, 10)
x_cols = list(set(feature_ls) & set(feature_ls2)) #include only feature columns which appear in both testing/training

#get predicted scores from top model
y_test_predicted = pipeline.build_models(test[x_cols], test[y_col], train[x_cols], train[y_col], [top_model['type']], top_model['parameters'])

#add predicted scores to testing
test['pred_score'] = y_test_predicted

#get 5% of testing data with highest predicted score
fifth_percent_index = test.sort_values('pred_score', ascending=False).index[(int(test.shape[0]*0.05))]
fifth_percent_pred_score = test.loc[fifth_percent_index]['pred_score']
top_five_pred_scores = test[test['pred_score'] > fifth_percent_pred_score].copy()

#add clusters
top_five_pred_scores = pipeline.create_clusters(top_five_pred_scores, feature_ls, 5)

#explore clusters
pipeline.explore_clusters_2(top_five_pred_scores, feature_ls, y_col, 0.2)

