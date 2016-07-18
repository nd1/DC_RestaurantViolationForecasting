'''
Prepare 311, SeeClickFix, Construction, and Crime data for use with main DataFrame and append the data. Write the final data frame to csv.

Fields pulled for 311/SeeClickFix data are documented in the documentation folder.

20160630 Nicole Donnelly
'''

import pandas as pd
import pymongo
import datetime
import time
import geopy
from geopy.distance import vincenty

def clean_master(df):
    #clean some items from sqlite exported data
    df.insp_date = pd.to_datetime(df.insp_date).dt.date
    df.license_number.fillna(0, inplace=True)
    df.license_number = df.license_number.astype(int)
    df.drop_duplicates(inplace=True)
    return df

def seeclickfix():
    #pull seeclickfix data from mongodb

    connection = pymongo.MongoClient()
    db = connection.seeclickfix
    input_data = db.see_click_fix_records
    data = pd.DataFrame(list(input_data.find()))

    service_records = ['Alley Cleaning', 'Bulk Collection', 'Container Removal', 'Dead Animal Collection', 'DOEE - Bag Law Tips', 'DOEE - Ban on Foam Food Containers', 'DOEE - Construction \xe2\x80\x93 Erosion Runoff', 'DOEE - General Environmental Concerns', 'DOEE - Nuisance Odor Complaints', 'Emergency - Power Outage/Wires Down', 'Grass and Weeds Mowing', 'Illegal Dumping', 'Insects', 'Public Space Litter Can-Collection', 'Public Space Litter Can- Installation/Removal/Repair', 'Recycling Cart - Repair', 'Recycling Collection - Missed', 'Rodent Inspection and Treatment', 'Sanitation Enforcement', 'Street Cleaning', 'Supercan - Repair', 'Trash Cart Repair', 'Trash Collection - Missed', 'Vacant Lot', 'Wire Down/Power Outage', 'DOEE - Construction Erosion Runoff', 'General Environmental Concerns (DOEE)', 'Litter Can - Installation ', 'Litter Can - Collection', 'Grass & Weeds Mowing', 'Residential Bulk Collection', 'Construction Site Environmental Concern / Tips (DDOE)', 'Residential bulk collection']

    #use only desired records
    data = data[data.summary.isin(service_records)]
    #use only specific columns
    data = data[['created_at', 'lng', 'lat', 'summary']]
    #change the column names
    col_names = ['date', 'lng', 'lat', 'issue']
    data.columns = col_names
    #convert the date object to datetime, drop the time info
    data.date = pd.to_datetime(data.date).dt.date
    #pull only date for required timeframe
    data = data[data.date >= datetime.date(year=2012,month=12,day=29)]
    #group the data by lat/ lng
    data = data.groupby(['lat', 'lng', 'date'])['issue'].count()
    data = data.reset_index()
    #create a coordinate column for computing distances
    data['coord'] = zip(data.lat, data.lng)
    return data

def crimes():
    #read in the crime data
    crime = pd.read_csv('Crime_Incidents.csv')
    #use only specific columns
    crime = crime[['REPORTDATETIME', 'X', 'Y', 'OFFENSE']]
    #convert the date object to datetime, drop the time info
    crime.REPORTDATETIME = pd.to_datetime(crime.REPORTDATETIME).dt.date
    #pull only date for required timeframe
    crime = crime[crime.REPORTDATETIME >= datetime.date(year=2012,month=12,day=29)]
    #group the data by lat/ lng
    crime = crime.groupby(['X', 'Y', 'REPORTDATETIME'])['OFFENSE'].count()
    crime = crime.reset_index()
    #create a coordinate column for computing distances
    crime['coord'] = zip(crime.Y, crime.X)
    return crime

def construction():
    #read in the construction data
    construction = pd.read_csv('Construction_Permits_via_DDOT_TOPs.csv')
    #use only specific columns
    consrtuction = construction[['EffectiveDate', 'ExpirationDate', 'X', 'Y', 'PermitType']]
    #a time with the year 3029 was discovered, drop it
    construction = construction[construction.EffectiveDate != construction.EffectiveDate.max()]
    #convert the date objects to datetime, drop the time info
    construction.EffectiveDate = pd.to_datetime(construction.EffectiveDate).dt.date
    construction.ExpirationDate = pd.to_datetime(construction.ExpirationDate).dt.date
    #pull only date for required timeframe
    construction = construction[(construction.EffectiveDate >= datetime.date(year=2012,month=12,day=29))]
    #group the data by lat/ lng
    construction = construction.groupby(['X', 'Y', 'EffectiveDate', 'ExpirationDate'])['PermitType'].count()
    construction = construction.reset_index()
    #create a coordinate column for computing distances
    construction['coord'] = zip(construction.Y, construction.X)
    return construction

def weather():
    #read in weather data
    weather = pd.read_csv('weather_data_edit.csv')
    #convert the date
    weather.date = pd.to_datetime(weather.date).dt.date
    return weather

def crime_count(lat, lng, date, df):
    #get the sum of crime reports for the 3 days before the inspection within 1 km of the location

    #restaurant location
    restaurant = (lat, lng)
    #time period
    start = date - datetime.timedelta(days=3)
    #pull the dates of interest from the dataframe
    df = df[(df.REPORTDATETIME >= start) & (df.REPORTDATETIME < date)]
    #compute the distance from the restaurant of reports in the date range
    df['distance'] = [vincenty(restaurant, x).km for x in df.coord]
    #keep only the reports that occured within 1 km
    df = df[df.distance < 1.0]
    #return the sum of the crime reports
    return df.OFFENSE.sum()

def service_counts(lat, lng, date, df):
    #get the sum of 311 service requests for the 3 days before the inspection within 1 km of the location

    #restaurant location
    restaurant = (lat, lng)
    #time period
    start = date - datetime.timedelta(days=3)
    #pull the dates of interest from the dataframe
    df = df[(df['date'] >= start) & (df['date'] <= date)]
    #compute the distance from the restaurant of reports in the date range
    df['distance'] = [vincenty(restaurant, x).km for x in df.coord]
    #keep only the reports that occured within 1 km
    df = df[df.distance < 1.0]
    #return the sum of the service requests
    return df.issue.sum()

def construction_count(lat, lng, date, df):
    #get the sum of active construction permits for the 3 days before the inspection within 1 km of the location

    #restaurant location
    restaurant = (lat, lng)
    #time period
    start = date - datetime.timedelta(days=3)
    #pull the dates of interest from the dataframe
    df = df[(df.EffectiveDate >= start) & (df.EffectiveDate <= date)]
    #compute the distance from the restaurant of reports in the date range
    df['distance'] = [vincenty(restaurant, x).km for x in df.coord]
    #keep only the reports that occured within 1 km
    df = df[df.distance < 1.0]
    #return the number of active construction permits
    return df.PermitType.sum()

def get_weather(date, df):
    #get the 3 day average high temperature for the report date
    temp = df['3-day_avg_maxTemp'][df.date == date]
    return temp.iloc[0]

def meld_dfs(df, crime, service, construction, weather):
    #add the crime, 311, and construction data to the master lsit
    start_time = time.time()
    print start_time
    #for each row in the master list, get the counts for crime reports, 311 service requests, and active construction permits. append them to the dataframe
    df['crime_count'] = df.apply(lambda row: crime_count(row.lat, row.lon, row.insp_date, crime), axis=1)
    df['311_count'] = df.apply(lambda row: service_counts(row.lat, row.lon, row.insp_date, service), axis=1)
    df['construction_count'] = df.apply(lambda row: construction_count(row.lat, row.lon, row.insp_date, construction), axis=1)
    df['avg_high_temp'] = df.apply(lambda row: get_weather(row.insp_date, weather), axis=1)

    print("--- %s seconds ---" % (time.time() - start_time))
    return df

def main():
    #load the master business list with inspection and yelp data
    master = pd.read_csv('master_03.csv')
    master = clean_master(master)
    #load the crime data
    df_crime = crimes()
    #load the service request data
    df_311 = seeclickfix()
    #load the construction data
    df_construction = construction()
    #load the weather data
    df_weather = weather()
    #combine the data sources and write to csv
    final = meld_dfs(master, df_crime, df_311, df_construction, df_weather)
    final.to_csv('multiple_sources_dataset.csv', index=False)

if __name__ == "__main__":
        main()
