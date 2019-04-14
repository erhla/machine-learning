import pipeline

df = pipeline.read_load('/home/erhla/Downloads/credit-data.csv')
pipeline.explore(df, ['PersonID','zipcode']) 
df = pipeline.preprocess(df, ['PersonID','zipcode']) 
df = pipeline.generate_features(df, 'SeriousDlqin2yrs', 'dummy')
df = pipeline.generate_features(df, 'MonthlyIncome', 'discretized', 10)

#categorical columns
df['zipcode'] = df['zipcode'].astype('category')

model, x_test, y_test = pipeline.build_classifier(df, 'SeriousDlqin2yrs', 0.2, 10, 5)
pipeline.evaluate_classifier(model, x_test, y_test)