'''
Homework 1
Eric Langowski
'''

import pandas as pd
from sodapy import Socrata
import requests

#Problem 1
CRIMES_18 = '3i3m-jwuy'
CRIMES_17 = 'd62x-nvdr'

def get_crime_data(id):
	client = Socrata('data.cityofchicago.org', 'HOfdnr49zcHGX5cuvxwbyTJjn',
		username='langowski@uchicago.edu',password='GCpGLLQ3EmJiNDP')
	current = 0
	results = client.get(id, limit=2000, offset=current)
	while len(results) > 0:
		current += 2000
		new = client.get(id, limit=2000, offset=current)
		if new:
			results.append(new)
		else:
			break
		print(current)
	return pd.DataFrame.from_records(results)

def download_17_18():
	crime_17 = get_crime_data(CRIMES_17)
	crime_18 = get_crime_data(CRIMES_18)
	return pd.concat(crime_17, crime_18)

CSV_URL = 'https://data.cityofchicago.org/api/views/'
CSV_FRAG = '/rows.csv?accessType=DOWNLOAD'
URLS = {'17': CSV_URL + CRIMES_17 + CSV_FRAG,
		'18': CSV_URL + CRIMES_18 + CSV_FRAG}

def download_17_18_alt():
	crime_17 = pd.read_csv(URLS['17'])
	crime_18 = pd.read_csv(URLS['18'])
	return pd.concat([crime_17, crime_18])


