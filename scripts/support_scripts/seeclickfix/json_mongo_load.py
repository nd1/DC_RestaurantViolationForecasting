'''
Dumps the issues in the json file returned fom the SeeClickFix api for easier loading to MongoDB. Then loads the files to one collection in mongo.

20160618 Nicole Donnelly
'''

import json
import os
import pymongo
from glob import iglob
from pprint import pprint

def json_edit():
    json_list = os.listdir('/Users/nicole/Desktop/GA/DC_DSI_capstone/data/SeeClickFix/api_json/')

    for json_file in json_list:
        root = '/Users/nicole/Desktop/GA/DC_DSI_capstone/data/SeeClickFix'
        in_path = os.path.join(root + '/api_json/' + json_file)
        out_path = os.path.join(root + '/edited_json/' + json_file)

        with open(in_path) as data_file:
            json_file = json.load(data_file)

        with open(out_path, 'w') as outfile:
            json.dump(json_file["issues"], outfile)

def json_mongo_load():
    json_edit()
    json_path = iglob(os.path.expanduser('~/Desktop/GA/DC_DSI_capstone/data/SeeClickFix/edited_json/*.json'))
    conn=pymongo.MongoClient()
    db = conn.seeclickfix
    for json_file in json_path:
        with open(json_file) as jf:
            see_click_fix = json.load(jf)
            for x in see_click_fix:
                db.see_click_fix_records.insert_one(x)

if __name__ == '__main__':
    json_mongo_load()
