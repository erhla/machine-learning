# -*- coding: utf-8 -*-
import pipeline
import pandas as pd

df = pipeline.read_load('/Users/erhla/Downloads/projects_2012_2013.csv')

#convert columns to datetime and add outcome column
df['date_posted'] = pd.to_datetime(df['date_posted'])
df['datefullyfunded'] = pd.to_datetime(df['datefullyfunded'])
df['days_to_fund'] = df['datefullyfunded'] - df['date_posted']
df['funded_within_60_days'] = pd.get_dummies(df['days_to_fund'] <= pd.Timedelta('60 days'), drop_first=True)

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

df = pipeline.preprocess(df, cols_to_fill, cols_to_drop_nas)
df, feature_ls = pipeline.generate_features(df, feature_dict, 10)

df = pipeline.create_clusters(df, feature_ls, 5)
pipeline.explore_clusters(df, feature_ls, y_col, 0.2)
df = pipeline.merge_cluster(df, [3, 4])
df = pipeline.split_cluster(df, feature_ls, 2, 3)


