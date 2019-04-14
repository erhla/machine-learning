'''
pipeline.py is a machine learning pipeline
Eric Langowski
version 0.0
'''
import numpy as np
import pandas as pd
from scipy import stats

#read/load data
def read_load(file_path):
	return pd.read_csv(file_path)

#explore data
def explore(df, exclude=''):
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

	for col_name in df.columns: #check for outliers
		current = df[col_name].dropna()
		outliers = current[(np.abs(stats.zscore(current)) > 5)] #find values 4sds
		if not outliers.empty:
			print(col_name, 'has possible outliers', outliers.shape[0], '\n')

#pre-process data
def preprocess(df, skipcols=''):
	for col_name in df.columns:
		if col_name not in skipcols:
			fill_val = df[col_name].agg('median')
			df[col_name] = df[col_name].fillna(fill_val)
			print('nas filled for', col_name)
	return df
