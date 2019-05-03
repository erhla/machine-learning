import pipeline
import pandas as pd

df = pipeline.read_load('/Users/erhla/Downloads/projects_2012_2013.csv')

df['date_posted'] = pd.to_datetime(df['date_posted'])
df['datefullyfunded'] = pd.to_datetime(df['datefullyfunded'])
df['days_to_fund'] = df['datefullyfunded'] - df['date_posted']
df['funded_within_60_days'] = pd.get_dummies(df['days_to_fund'] <= pd.Timedelta('60 days'), drop_first=True)

df = pipeline.preprocess(df, ['students_reached'], ['primary_focus_area', 'resource_type', 'grade_level'])

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
df, feature_ls = pipeline.generate_features(df, feature_dict, 10)
pipeline.build_models(df, feature_ls, 'funded_within_60_days', ['DT'])
