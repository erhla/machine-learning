import pipeline
import pandas as pd

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
total.to_csv('example.csv')
    
