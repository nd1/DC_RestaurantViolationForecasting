'''
Parse the local health inspection report files and write the results to csv.

The purpose of this script is to pull the license/ customer information from the health inspection reports since it was discovered the permit_id number used by DOH appears to be a unique identifier separate from actual business license number.

20160628 Nicole Donnelly
'''

import os
import re
import urllib
import pandas as pd
from bs4 import BeautifulSoup

def clean_license_result():

    df = pd.read_csv('license_result.csv')

    #clean the report_id
    df.report_id.replace(regex=True, to_replace='.html', value='', inplace=True)

    #clean the license number
    df.license_number.replace(regex=True, to_replace='\n\n\nLicense/Customer No.\n\t\t\t\t\t\t\t\t\xc2\xa0', value='', inplace=True)
    df.license_number.replace(regex=True, to_replace='\n\t\t\t\t\t\t\t\n\n', value='', inplace=True)

    df.license_number.replace(regex=True, to_replace='[xX]*-', value='', inplace=True)

    df.to_csv('license_info.csv', encoding='utf-8', index=False)

def parse_reports():
    file_name = []
    license = []
    errors = []

    report_list = os.listdir('/Users/nicole/Desktop/GA/DC_DSI_capstone/health_inspection_reports/restaurant_total/')
    #print len(report_list)

    for report in report_list:
        #print "in for loop"
        #soup the file
        html_file = '/Users/nicole/Desktop/GA/DC_DSI_capstone/health_inspection_reports/restaurant_total/' + report
        html = urllib.urlopen(html_file).read()
        soup = BeautifulSoup(html, 'html.parser')

        #inspeciton info
        inspection = soup.find_all('tr')
        #print "got inspection, len %d" % len(inspection)
        if len(inspection) == 0:
            #print "in the if"
            #print "Error! Cannot parse %s" % report_test
            errors.append(report)
        else:
            print report
            #add the report number we are looking at
            file_name.append(report)
            license.append(inspection[8].get_text())

    #write the report info to csv
    #print "at the write"
    license_result = pd.DataFrame({'report_id': file_name, 'license_number': license})

    license_result.to_csv('license_result.csv', encoding='utf-8', index=False)
    #write the error list to a file
    with open('errors.txt', 'w') as f:
        for item in errors:
            f.write("{}\n".format(item))

if __name__ == "__main__":
    #parse_reports()
    clean_license_result()
