'''
Use the GooglePlaces API to get phone numbers for business we need Yelp Ids for.

original unknown phone_num- 1011 reduced to 264 with type and address added, then just address added.

can be run with or without types=[types.TYPE_FOOD] (deprectated parameter, support will stop in 2017) or address. I added these after an initial pass.

20160624 Nicole Donnelly
'''

from googleplaces import GooglePlaces, types, lang
import pandas as pd
import numpy as np

def google_places():
    api_key = 'add api key'
    geo = pd.read_csv('updated_master3.csv')
    google_places = GooglePlaces(api_key)

    for idx in geo.index:
        if ((type(geo.phone_num[idx]) != str) and (type(geo.YelpID[idx]) != str)):
            print geo.name[idx]
            lat = geo.lat[idx]
            lon = geo.lon[idx]
            result = google_places.nearby_search(lat_lng={'lat': lat, 'lng':lon}, rankby='distance', name=geo.name[idx].decode('utf-8'), keyword=geo.address[idx])
            #result = google_places.nearby_search(lat_lng={'lat': lat, 'lng':lon}, rankby='distance', name=geo.name[idx].decode('utf-8'), types=[types.TYPE_FOOD])
            #result = google_places.nearby_search(lat_lng={'lat': lat, 'lng':lon}, rankby='distance', name=geo.name[idx].decode('utf-8'), keyword=geo.address[idx], types=[types.TYPE_FOOD])
            if len(result.places) == 1:
                x = result.places[0]
                x.get_details()
                geo.set_value(idx, 'phone_num', x.local_phone_number)
                print "updated %s" % geo.name[idx]
            elif len(result.places) > 1:
                for place in result.places:
                    if (float(place.geo_location['lat']) == lat and float(place.geo_location['lng']) == lon):
                        x = place
                        x.get_details()
                        geo.set_value(idx, 'phone_num', x.local_phone_number)
                        print "updated %s" % geo.name[idx]
            else:
                print "for %s, length is %d" % (geo.name[idx], len(result.places))

    geo.phone_num.replace(regex=True, to_replace='\(|\)|-| ', value='', inplace=True)
    geo.to_csv('updated_master4.csv', index=False)
if __name__ == "__main__":
    google_places()
