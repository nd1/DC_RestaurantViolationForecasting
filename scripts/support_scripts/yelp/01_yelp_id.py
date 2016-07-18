import json
import os
import pandas as pd
from yelpapi import YelpAPI

'''attempt to identify yelp business id

Yelp API limitation- businesses with no reviews are not returned so if the business is closed it will not be in your results.

DC Data limitation- there are building cafeteria's and hotel kitchens in the Restaurant Total category that are not necessarily "public" restaurants.

This script uses the general Yelp search api to return potential matches. You can then decide whether it matches. If it does, you can update the dataframe.

After using this script I ran it one more time against yelp-master.csv adding the "c" and "x" options and removing the name from the search.

Future-- add error handling.

20160624 Nicole Donnelly
'''

#define yelp credential variables
consumer_key = 'add key'
consumer_secret = 'add key'
token = 'add key'
token_secret = 'add key'

#create a yelp api instance
yelp_api = YelpAPI(consumer_key, consumer_secret, token, token_secret)

def here_goes():
    #data = pd.read_csv('yelp_master.csv', sep='\t')
    data = pd.read_csv('yelp_update.csv')
    data.name.replace(regex=True, to_replace='\(.?\)', value='', inplace=True)

    for index, row in data.iterrows():
        #if (type(row.phone_num) != str and type(row.YelpID) != str):
        if (type(row.YelpID) != str):
            i = 0
            os.system('clear')
            print index
            print row['name'], row.address, '\n'
            response = yelp_api.search_query(term=row['name'], location=row['address'], limit=20, radius=400, sort=1)
            #response = yelp_api.search_query(location=row['address'], limit=20, radius=400, sort=1)
            while i < len(response['businesses']):
                print response['businesses'][i]['id']
                print response['businesses'][i]['location'], '\n'
                input = raw_input('Is this a match? ')
                if input == 'y':
                    print 'yay\n'
                    data.set_value(index, 'YelpID', response['businesses'][i]['id'])
                    i = len(response['businesses'])
                    os.system('clear')
                elif input == 'q':
                    i = len(response['businesses'])
                elif input == 'c':
                    data.set_value(index, 'YelpID', 'closed')
                    i = len(response['businesses'])
                elif input == 'x':
                    data.set_value(index, 'YelpID', 'no_yelp')
                else:
                    print'boo\n'
                    i += 1
    data.to_csv('yelp_master3.csv', index=False, encoding='utf-8')
if __name__ == "__main__":
    here_goes()
