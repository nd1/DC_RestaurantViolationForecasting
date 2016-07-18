'''
Parse the local health inspection report files and writ the results to csv.

report_result will contain the following information from the report:
Date of inspection; time in; time out; phone number; inspector name; inspector badge; risk category; report id

violation_list will contain:
the report id, each violation from the report, whether the report was corrected on site, and whether it was a repeat violation as a single line. Check the final report for a report violation value of 3-- this indicates the report info needs to be manually validated because the inspection table in the report did not have the expected number of elements.

Each csv is then cleaned and a final version is saved.

Yes, this code is fairly long and should be cleaned up. This is a prototype for now.

20160621 Nicole Donnelly
'''

import os
import re
import urllib
import pandas as pd
from bs4 import BeautifulSoup

def clean_report_result():

    df = pd.read_csv('report_result.csv')

    #clean the inspector name badge number
    df.insp_name.replace(regex=True, to_replace='<td style="width:225px; vertical-align: bottom;">', value='', inplace=True)
    df.insp_name.replace(regex=True, to_replace='\xc2\xa0', value='', inplace=True)
    df.insp_name.replace(regex=True, to_replace='</td>', value='', inplace=True)
    df.insp_badge.replace(regex=True, to_replace='<td style="width:90px; vertical-align: bottom;">|\xc2\xa0</td>', value='', inplace=True)

    #clean the inspection details
    df.insp_det.replace(regex=True, to_replace='\n*Date of Inspection', value='', inplace=True)
    df.insp_det.replace(regex=True, to_replace='\n', value='', inplace=True)
    df.insp_det.replace(regex=True, to_replace='\xc2\xa0', value='', inplace=True)

    #clean the phone number
    df.phone_num.replace(regex=True, to_replace='\n\n\nTelephone\n\xc2\xa0', value='', inplace=True)
    df.phone_num.replace(regex=True, to_replace='\n\xc2\xa0E-mail address\n\t\t\t\t\t\t\t\t\xc2\xa0.*\n\t\t\t\t\t\t\t\n\n', value='', inplace=True)
    df.phone_num = [re.sub(r'[^\w]', '', x).strip() for x in df.phone_num]

    #clean the report_id
    df.report_id.replace(regex=True, to_replace='.html', value='', inplace=True)

    #split the inspection details data into separate columns
    for index, row in df.iterrows():
        txt_split = df.insp_det[index].split()
        df.set_value(index, 'insp_date', txt_split[0])
        df.set_value(index, 'insp_timein', txt_split[1])
        df.set_value(index, 'insp_timeout', txt_split[2])

    #clean the new columns and drop original data
    df.insp_date.replace(regex=True, to_replace='Time', value='', inplace=True)
    df.insp_timein.replace(regex=True, to_replace='Time', value='', inplace=True)
    df.insp_timein.replace(regex=True, to_replace='In', value='', inplace=True)
    df.insp_timeout.replace(regex=True, to_replace='Out', value='', inplace=True)
    df.drop('insp_det', axis=1, inplace=True)

    #change the column types
    df.insp_date = pd.to_datetime(df.insp_date)
    df.insp_timein = pd.to_datetime(df.insp_timein, errors='coerce').dt.time
    df.insp_timeout = pd.to_datetime(df.insp_timeout, errors='coerce').dt.time

    #reorder the column names and write to csv
    new_order = ['report_id', 'risk', 'phone_num', 'insp_date', 'insp_timein', 'insp_timeout', 'insp_name', 'insp_badge']
    cleaned = df[new_order]
    cleaned.to_csv('cleaned_report_results.csv', encoding='utf-8', index=False)

def clean_violation_list():
    df = pd.read_csv('violation_list.csv')

    #clean report_id
    df.report.replace(regex=True, to_replace='.html', value='', inplace=True)

    #reorder and write to csv
    new_order = ['report', 'violation', 'corrected_on_site', 'repeat_violation']
    cleaned = df[new_order]
    cleaned.to_csv('cleaned_violation_list.csv', encoding='utf-8', index=False)

def parse_reports():
    phone=[]
    insp_det=[]
    risk=[]
    insp_name=[]
    insp_badge=[]
    file_name=[]
    report_id = []
    violation_id =[]
    cos = []
    r_viol = []
    errors =[]
    skip_list = []

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
            #print "in the else"
            #add the report number we are looking at
            file_name.append(report)
            phone.append(inspection[5].get_text())
            insp_det.append(inspection[6].get_text())

            #identify the risk category by finding which box was checked
            #if no box checked, list risk category as 0
            risk_num = 0
            for element in inspection[10].find_all('div'):
                #print "finding risk"
                if re.search('FF0000', str(element)):
                    risk.append(risk_num)
                    risk_num = 0
                else:
                    if risk_num < 5:
                        risk_num +=1
                    else:
                        risk.append(0)

            tables = soup.find_all('table')

            #find inspector details
            #print "inspector"
            inspector_info = tables[11]
            inspector = inspector_info.find_all("td")
            insp_name.append(inspector[1])
            insp_badge.append(inspector[2])

            #pull the violations info
            #add a 0 row for the report id to catch no violations
            report_id.append(report)
            violation_id.append(0)
            cos.append(0)
            r_viol.append(0)
            #check the length of inspection. when the violation table exists, it will be greater than 100
            if len(inspection) > 100:
                for i in range(20,93):
                    #create a row for each violation
                    if len(inspection[i].find_all('td')) == 8:
                        if re.search('CC0000', str(inspection[i].find_all('td')[1])):
                            #out of compliance
                            report_id.append(report)
                            violation_id.append(inspection[i].find_all('td')[4].get_text())
                            if re.search('000000', str(inspection[i].find_all('td')[6])):
                                    #corrected on site
                                    cos.append(1)
                            else:
                                    cos.append(0)

                            if re.search('000000', str(inspection[i].find_all('td')[7])):
                                    #repeat violation
                                    r_viol.append(1)
                            else:
                                    r_viol.append(0)
                    elif len(inspection[i].find_all('td')) == 7:
                        #the one report where I found this did not have two items to account for "corrected on site" and "repeat violation". Checking the one item to set corrected on site, will add 3 to report violation in that case to flag this for manual checking.
                        if re.search('CC0000', str(inspection[i].find_all('td')[1])):
                            #out of compliance
                            report_id.append(report)
                            violation_id.append(inspection[i].find_all('td')[4].get_text())
                            if re.search('000000', str(inspection[i].find_all('td')[6])):
                                    #corrected on site
                                    cos.append(1)
                            else:
                                    cos.append(0)
                            r_viol.append(3)

                    #this code is to catch any formatting changes to the violations table
                    elif len(inspection[i].find_all('td')) == 3:
                          pass
                    elif len(inspection[i].find_all('td')) == 1:
                          pass
                    else:
                          print report, " unexpected element length x[%d]" % i
                print report, " complete"
            #check lengths of lists. if any are not pulling necessary data, code will break so report of issue is known and can be manually inspected and this script can be updated to accomodate variation.
            x = len(file_name)
            if len(phone) < x:
                print "phone fails"
                break
            elif len(insp_det) < x:
                print "insp_det"
                break
            elif len(risk) < x:
                print "risk fails"
                break
            elif len(insp_name) < x:
                print "insp_name fails"
                break
            elif len(insp_badge) < x:
                print "insp_badge fails"
                break

    #write the report info to csv
    #print "at the write"
    report_result = pd.DataFrame({'report_id': file_name, 'phone_num': phone, 'insp_det': insp_det, 'risk': risk, 'insp_name': insp_name, 'insp_badge': insp_badge})
    report_result.to_csv('report_result.csv', encoding='utf-8', index=False)

    #write the violation list
    violation_list = pd.DataFrame({'report': report_id, 'violation': violation_id, 'corrected_on_site': cos, 'repeat_violation':r_viol})
    violation_list.to_csv('violation_list.csv', encoding='utf-8', index=False)

    #write the error list to a file
    with open('errors.txt', 'w') as f:
        for item in errors:
            f.write("{}\n".format(item))

if __name__ == "__main__":
    parse_reports()
    clean_report_result()
    clean_violation_list()
