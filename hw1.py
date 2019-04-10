'''
Homework 1
Eric Langowski
'''
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
	client = Socrata('data.cityofchicago.org', None)
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

census_variables = {"S0101_C01_001E": "Population",
					"S1701_C02_001E": "Population in Poverty",
					"S1901_C01_012E": "Median income",
					"S1501_C02_015E": "Percent Population with Bachelors",
					"S2303_C02_033E": "Percent Population working full-time",
					"S1601_C02_003E": "Percent speaking not English"}

census_url = "https://api.census.gov/data/2017/acs/acs5/subject?get="
census_frag = "&for=tract:*&in=state:17&in=county:031"

def get_census_data():
	url = census_url + ",".join(census_variables.keys()) + census_frag
	df = pd.read_json(url, orient='records')
	new_cols = list(census_variables.values()) + ['state', 'county', 'tract']
	df.columns = new_cols
	df = df.drop(df.index[0])
	return df

def geomerge(crime_data):
	#Some of this code is from my group's 122 project
	#which made a spatial join on Chicago's neighborhoods

	client = Socrata('data.cityofchicago.org', None)
	files = client.get(CENSUS_TRACTS)
	df = pd.DataFrame(files)
	df['the_geom'] = df['the_geom'].apply(shapely.geometry.shape)
	df = geopandas.GeoDataFrame(df, geometry='the_geom')
	df.crs = {'init': 'espg:4326'}
	
	crime_data = crime_data[crime_data['Longitude'].notna()]
	crime_data['Location'] = crime_data['Location'].str.strip('()').str.split(',')    
	crime_data['Location'] = crime_data['Location'].apply(lambda x: (float(x[1]), float(x[0])))#tuple(map(float,x)))     
	crime_data.loc[:, 'Location'] = crime_data['Location'].apply(shapely.geometry.Point)
	geodf = geopandas.GeoDataFrame(crime_data, geometry='Location')
	geodf.crs = {'init': 'espg:4326'}
	#geodf = geodf.to_crs({'init': 'espg:4326'})

	return geopandas.sjoin(geodf, df, how='left', op='within')

