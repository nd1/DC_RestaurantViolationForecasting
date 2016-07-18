'''
This script cleans and parses info in the csv created by insp_list_pages.py. The output will be used to download inspection reports.

Nicole Donnelly 201606
'''

import pandas as pd

def clean_insp_list(data):

    data.dropna(inplace=True)
    #remove spaces in column names
    data.columns = [x.strip().replace(' ', '') for x in df.columns]
    #remove extra characters so insp_url can be appended to a base
    data.insp_url.replace(regex=True, to_replace=r'\.\.', value=r'', inplace=True)

    #create inspec_id to have a column with just the inpesction report number
    data['inspec_id'] = data.insp_url.replace(regex=True, to_replace=r'/lib/mod/inspection/paper/_paper_food_inspection_report.cfm\?inspectionID=|&wguid=1367&wgunm=sysact&wgdmn=431', value=r'')

    #split the insp_text column
    for index, row in data.iterrows():

        text_split = row.insp_text.split()
        if len(text_split) == 1:
            data.set_value(index, 'insp_type', None)
            data.set_value(index, 'insp_date', None)
        elif len(text_split) == 5:
            insp_type = text_split[0]
            insp_date = text_split[2] + ' ' + text_split[3] + ' ' + text_split[4]
            data.set_value(index, 'insp_type', insp_type)
            data.set_value(index, 'insp_date', insp_date)
        elif len(text_split) == 6:
            insp_type = text_split[0] + ' ' + text_split[1]
            insp_date = text_split[3] + ' ' + text_split[4] + ' ' + text_split[5]
            data.set_value(index, 'insp_type', insp_type)
            data.set_value(index, 'insp_date', insp_date)
        elif len(text_split) == 7:
            insp_type = text_split[0] + ' ' + text_split[1] + ' ' + text_split[2]
            insp_date = text_split[4] + ' ' + text_split[5] + ' ' + text_split[6]
            data.set_value(index, 'insp_type', insp_type)
            data.set_value(index, 'insp_date', insp_date)
        else:
            print "Unexpected split length for index %d" % index

    data.drop('insp_text', axis=1, inplace=True)

    data.insp_type.replace(regex=True, to_replace=':', value='', inplace=True)

    #format the date column
    data.insp_date = pd.to_datetime(data.insp_date)
    data.to_csv('inpection_list.csv', index=False)
    print "Clean inspection data written to CSV."

if __name__ == "__main__":
    df = pd.read_csv('inspections_raw_rtotal.csv')
    clean_insp_list(df)
