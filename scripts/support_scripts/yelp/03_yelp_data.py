import json

import pandas as pd

from yelpapi import YelpAPI

''' Use yelp crosswalk file to pull yelp data.

Nicole Donnelly 31May2016'''

#define yelp credential variables
consumer_key = 'scj0gaCcuL7LBJlO-rL6wg'
consumer_secret = '056Y1SGdIjUYpdvBBKmApy6OOPE'
token = 'RGgZTmWBNgf5GQ5xda2R8uz2Udwmttla'
token_secret = 'yBTwct5U9oqDSNgK_NQcuCav3ew'

#create a yelp api instance
yelp_api = YelpAPI(consumer_key, consumer_secret, token, token_secret)


def yelp_pull(crosswalk_file):
    #using the yelp id in the crosswalk file, pull the business data
    #the yelpapi packaged returns the data as a dictionary, compile for dataframe

    yelp_ids = pd.read_csv(crosswalk_file, sep='\t')
    yelp_ids = yelp_ids[~yelp_ids.YelpID.isin(['no_yelp', 'closed'])]
    #create an empty dataframe to hold data
    yelp_data = pd.DataFrame(columns=['permitID', 'yelpID', 'categories', 'street_address', 'zip', 'lat', 'lon', 'rating', 'review_count'])

    skip_list=[]

    for index, row in yelp_ids.iterrows():
        try:
            if row.YelpID not in skip_list:
                yelpID = row.YelpID
                permitID = row.PermitID
                print permitID
                response = yelp_api.business_query(id=yelpID)
                yelp_data.loc[len(yelp_data)] = [permitID, response['id'], response['categories'], response['location']['address'], response['location']['postal_code'], response['location']['coordinate']['latitude'],response['location']['coordinate']['longitude'], response['rating'], response['review_count']]
            else:
                print row.YelpID, " in skip list"

        except YelpAPI.YelpAPIError as e:
                print yelpID, e
                skip_list.append(yelpID)

    print "loop fini"

    yelp_data.to_csv('yelp_data.csv', encoding='utf8', index=False)

if __name__ == "__main__":
    yelp_pull('yelp_crosswalk.csv')
