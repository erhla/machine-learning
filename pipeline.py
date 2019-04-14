'''
pipeline.py is a machine learning pipeline
Eric Langowski
version 0.0
'''
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import tree

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

	for col_name in df.columns: #check for outliers
		current = df[col_name].dropna()
		outliers = current[(np.abs(stats.zscore(current)) > 5)] #find values 5+ sds
		if not outliers.empty:
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
			fill_val = df[col_name].agg('median')
			df[col_name] = df[col_name].fillna(fill_val)
			print('nas filled for', col_name)
	return df

#generate features
def generate_features(df, col_name, feature_type, division_num=4):
	'''
	generate features creates new columns in the dataframe as features
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
		df[col_name + '_dis'] = pd.qcut(df[col_name], bins, labels=labels)
		print(col_name, 'discretize')
	if feature_type == 'dummy':
		unique = df[col_name].unique()
		print(col_name, 'has values: ', unique)
		df[col_name] = df[col_name].astype('category')
		if len(unique) != 2:
			dummies = pd.get_dummies(df[col_name], prefix=col_name)
			df[dummies.columns] = dummies
			print('target variable has more than two values, multiple dummies created')
		else:
			df[col_name] = df[col_name].cat.rename_categories([-1,1])
			print('dummy created for', col_name)
	return df

#build classifier
def build_classifier(df, outcome_col, test_size, max_depth, min_split_size):
	'''
	inspired by http://dataaspirant.com/2017/02/01/decision-tree-algorithm-python-with-scikit-learn/
	'''
	col_set = set(df.columns)
	col_set.remove(outcome_col)
	X, Y = df[list(col_set)], df[outcome_col]
	X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size)
	dt_tree = tree.DecisionTreeClassifier(max_depth=max_depth, min_samples_split=min_split_size)
	dt_tree.fit(X_train, Y_train)
	#to-do add tree visualization
	return dt_tree

#evaluate classifier
def evaluate_classifier(fitted_tree):
	pass
