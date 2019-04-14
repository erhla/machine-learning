import pipeline

df = pipeline.read_load('/home/erhla/Downloads/credit-data.csv')
pipeline.explore(df, ['PersonID','zipcode']) 
df = pipeline.preprocess(df, ['PersonID','zipcode']) 
df = pipeline.generate_features(df, 'SeriousDlqin2yrs', 'dummy')
df = pipeline.generate_features(df, 'MonthlyIncome', 'discretize', 10)

#categorical columns
df['zipcode'] = df['zipcode'].astype('category')
