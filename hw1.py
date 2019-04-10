'''
Homework 1
Eric Langowski
'''
#remove API key


import pandas as pd
from sodapy import Socrata
import requests
import geopandas
import shapely

#Problem 1
CRIMES_18 = '3i3m-jwuy'
CRIMES_17 = 'd62x-nvdr'
CENSUS_TRACTS = '74p9-q2aq'

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
	df = pd.concat([crime_17, crime_18])
	df['Date'] = pd.to_datetime(df['Date'])
	return df

col_ls = ["S0101_C01_001E","S1701_C02_001E","S1901_C01_012E","S1501_C02_015E","S2303_C02_033E","S1601_C02_003E"]
census_url = "https://api.census.gov/data/2017/acs/acs5/subject?get="
census_frag = "&for=tract:*&in=state:17&in=county:031"




def geomerge(census_data, crime_data):
	#Some of this code is from my group's 122 project
	#which made a spatial join on Chicago's neighborhoods

	client = Socrata('data.cityofchicago.org', None)
	files = client.get(CENSUS_TRACTS)
	df = pd.DataFrame(files)
	df['the_geom'] = df['the_geom'].apply(shapely.geometry.shape)
	df = geopandas.GeoDataFrame(df, geometry='the_geom')
	df.crs = {'init': 'espg:4326'}
	
	crime_data['coordinates'] = list(zip(df[lon], df[lat]))
	crime_data.loc[:, 'coordinates'] = crime_data['coordinates'].apply(shapely.geometry.Point)
	geodf = geopandas.GeoDataFrame(crime_data, geometry='coordinates')
	geodf.crs = {'init': 'espg:4326'}

	merged = geopandas.sjoin(crime_data, df, how='left', op='within')


	return df



def get_census_data():
	url = census_url + ",".join(col_ls) + census_frag
	df = pd.read_json(url, orient='records')
	df.columns = df.iloc[0]
	df = df.drop(df.index[0])
	return df
