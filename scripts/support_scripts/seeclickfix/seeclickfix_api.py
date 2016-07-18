'''Pull one page of 100 results from seeclickfix using the global PARAMS value if the parameters are not supplied.
If there are more than 100 results, make another pull passing paramters that include the next page to be pulled.
Nicole Donnelly 30May2016'''

import requests
import json

#used this code to make an api pull at 2222 2016 May 30 2016

def get_seeclickfix(page=1, pulled=0, search_params = {'place_url': 'district-of-columbia', 'after': '2013-04-01', 'per_page': 100}):

    #base_url for usajobs api to build the request url
    base_url = 'https://seeclickfix.com/api/v2/issues'

    #send a get request with the url, parameters, and header
    myResponse = requests.get(url = base_url, params = search_params)

    # For successful API call, response code will be 200 (OK)
    if(myResponse.ok):

        # Loading the response data into a dict variable
        data = json.loads(myResponse.content)

        #get the total search result count and set it to count_all. the API only allows 100 results per page
        count_all = data['metadata']['pagination']['entries']

        #track the number of items we have pulled with our requests
        pulled = pulled + 100

        #create a file name that reflects which page of results it contains and write that file
        file_name = '../data/SeeClickFix/data%d.json' % page
        with open(file_name, 'w') as outfile:
            json.dump(data, outfile)

        #check to see if we pulled all the results. If not, increment the page count, update the parameters dictionary to include the page number, and run the process again.
        if pulled < count_all:
            page += 1
            page_param = {'page': page}
            search_params.update(page_param)
            print search_params
            get_seeclickfix(page, pulled, search_params)

    else:
      # If response code is not ok (200), print the resulting http error code with description
        myResponse.raise_for_status()

if __name__ == '__main__':
    get_seeclickfix()
