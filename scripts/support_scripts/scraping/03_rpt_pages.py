from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait

import pandas as pd
import random
import time
import os

''' Pull copies of the inspection reports with the info compiled. It has been noted that scraping requests appear to lead to some kind of access restriction. Pull the full reports to local files to parse later.

items to add: error checking to make sure page downloaded.

Nicole Donnelly 20160614
'''

def rpt_pages():

    base_url= 'http://dc.healthinspections.us/webadmin/dhd_431'

    if not os.path.exists('health_inspection_reports'):
        os.makedirs('health_inspection_reports')

    df = pd.read_csv('inpection_list.csv')

    for index, row in df.iterrows():
        file_name = str(row.inspec_id) + '.html'

        if os.path.isfile(os.getcwd() + '/health_inspection_reports/' + file_name):
            print "File previously downloaded."
        elif row.insp_url == 'none':
            print "no reports for permit id %d" % row.permit
        else:
            seconds = 5 + (random.random() * 5)
            time.sleep(seconds)
            url = base_url + row.insp_url
            driver = webdriver.Chrome('/Users/nicole/Desktop/GA/DC_DSI_capstone/chromedriver')
            driver.get(url)
            outpath = os.getcwd() + '/health_inspection_reports/' +file_name
            print "pulling %r" % row.insp_url
            with open(outpath, 'w') as f:
                f.write(driver.page_source.encode('utf-8'))
            driver.close()

if __name__ == "__main__":
    rpt_pages()
