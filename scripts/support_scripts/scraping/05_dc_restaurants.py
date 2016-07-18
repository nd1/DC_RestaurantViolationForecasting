'''
Combines scraped health inspection data to create a master list of buisness in the "Restaurant Total" category. This will be used to identify Yelp ids for each business.

This also pulls in data from a csv available from http://opendatadc.org/dataset/restaurant-inspection-data The data at that location has not been updating. However, there is already a partial list of yelp ids that will be useful.

Once the list is updated, write it to csv. Use that csv to get lat and lon values. If the process fails, the file will be written with the data it has. This can then be used to run the code again and resume the process if needed.
These values will be used with the Google Places API to help pull phone numbers for the Yelp search.

Nicole Donnelly 20160623
'''

from geopy.geocoders import GoogleV3
import pandas as pd
import numpy as np

def get_latlon():
    master = pd.read_csv('master_list.csv')
    try:
    #use geopy to determinte lat/ lon of items
        geolocator = GoogleV3('api key')
        for idx in master.index:
            if 'lat' not in master:
                master['lat'] = np.nan
                master ['lon'] = np.nan
            if np.isnan(master.lat[idx]) and master.lon[idx]:
                location = geolocator.geocode(master.address[idx], timeout=5)
                if location is not None:
                    print idx
                    master.set_value(idx, 'lat', location.latitude)
                    master.set_value(idx, 'lon', location.longitude)
    except Exception,e:
        master.to_csv('geo_master.csv', index=False)
        print "csv written, closing due to error"
        print e

    master.to_csv('geo_master.csv', index=False)

def create_master():
    #import csvs to merge
    #report_id and phone_num
    rpt_list = pd.read_csv('cleaned_report_results.csv')

    #merge with rpt_list to get the phone_num based on inspection report
    insp_list = pd.read_csv('inpection_list_restauranttotal.csv')

    #primary business info
    bus_df = pd.read_csv('business_list.csv')

    #incomplete yelp id lsit for businesses available from Code For DC
    yelp = pd.read_csv('yelp_crosswalk.csv', sep='\t')

    #we only want to work on the items in the Restaurant Total category
    filter_list=['Restaurant Total']
    bus_df = bus_df.loc[bus_df['category'].isin(filter_list)]

    #merge rpt_list and insp_list, keep only the business permit id and phone number
    #drop duplicates based on the permit_id
    merged = pd.merge(left=insp_list, right=rpt_list, left_on='inspec_id', right_on='report_id')
    phone_list = merged[['permit_id', 'phone_num']]
    phone_list.drop_duplicates('permit_id', inplace=True)

    #merge the phone numbers with the business list
    master = pd.merge(left=bus_df, right=phone_list, how='left', left_on='permit_id', right_on='permit_id')

    #merge the known yelp ids and drop the extra permit column
    master = pd.merge(left=master, right=yelp, how='left', left_on='permit_id', right_on='PermitID')
    master.drop('PermitID', axis=1, inplace=True)

    #write to csv
    master.to_csv('master_list.csv', index=False)

if __name__ == '__main__':
    create_master()
    #get_latlon()
