'''
Determine yelp id for businesses by phone number.

Use the last file written with the yelp_id.py script. Run yelp_id again on the results to pick up itmes that have a phone number but no yelp id. Then manually reconcile.

20160625 Nicole Donnelly
'''

import json
import re
import pandas as pd
from yelpapi import YelpAPI

#define yelp credential variables
consumer_key = 'add key'
consumer_secret = 'add key'
token = 'add key'
token_secret = 'add key'

#create a yelp api instance
yelp_api = YelpAPI(consumer_key, consumer_secret, token, token_secret)

def yelp_phone_search():
    data = pd.read_csv('yelp_master2.csv', sep='\t')
    for index, row in data.iterrows():
        if (type(row.YelpID) != str):
            if re.search('[a-zA-Z]', row.phone_num):
                print "Phone number for ", row.name, "contains letters."
            else:
                response = yelp_api.phone_search_query(phone=('+1' + row.phone_num))
                if response['total'] > 0:
                    data.set_value(index, 'YelpID', (response['businesses'][0]['id']).encode('utf-8'))
                print "updated ", row.name
    data.to_csv('yelp_update.csv', index=False)

if __name__ == "__main__":
    yelp_phone_search()
