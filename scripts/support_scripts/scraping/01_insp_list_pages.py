'''
This script assumes you have a file called business_list.csv built by dc_health.py available in the same directory.

This script will filter by business type. You can either change the filters or edit the csv directly to create the dataset you want to pull inspection reports for.

Due to issues encountered while scraping, the script then pulls each report to a local html file.

Nicole Donnelly 20160608
with inital code from Kate Rabinowitz
'''

import csv
import os
import random
import time

from selenium import webdriver

def insp_list_pages():
    if not os.path.exists('permit_pages'):
        os.makedirs('permit_pages')

    data = pd.read_csv('business_list.csv')
    filter_list = ['Restaurant Total']
    data = data.loc[data['category'].isin(filter_list)]

    for index, row in data.iterrows():
        url = row.permit_url
        driver = webdriver.Chrome("/Users/nicole/Desktop/GA/DC_DSI_capstone/chromedriver")
        driver.get(url)
        seconds = 5 + (random.random() * 5)
        time.sleep(seconds)
        file_name = str(row.permit_id) + '.html'
        outpath = os.getcwd() + '/permit_pages/' +file_name
        print "pulling %r" % row.permit_id
        with open(outpath, 'w') as f:
            f.write(driver.page_source.encode('utf-8'))
        driver.close()

if __name__ == "__main__":
    insp_list_pages()
