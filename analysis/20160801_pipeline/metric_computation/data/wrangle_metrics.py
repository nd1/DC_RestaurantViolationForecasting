'''
Using the dataset created from multiple sources, perform final feature extraction and wrangling to produce datasets for metrics computation.

We will use data from 3/16-5/16 for the metrics.

20160731 Nicole Donnelly
'''

import datetime
import numpy as np
import os
import pandas as pd
import warnings

def time_diff_computation(X):
    '''
    Convert date values to date, sort the dataframe and use groupby to put the instances together for computation. Copy the insp_date to orig_insp_date. Then for dates later than 3/1/16, set them to 3/1/16. The earliest inspection will end up with NaT in the time_diff column. Set that time difference to 0.
    '''

    X.insp_date = pd.to_datetime(X.insp_date).dt.date
    X['orig_insp_date'] = X['insp_date']

    X.sort_values(by=['yelp_id', 'insp_date'], ascending=[True, True], inplace=True)

    X.insp_date = [datetime.date(year=2016,month=3,day=1) if x >= datetime.date(year=2016,month=3,day=1) else x for x in X.insp_date]

    X['time_diff'] = X.groupby('yelp_id', sort=False)['insp_date'].diff()

    X.loc[X.time_diff.isnull(), 'time_diff'] = 0
    X.time_diff = [x.days for x in X.time_diff]
    return X

def prev_viol(X):
    '''
    Add the count of previous violtions on the last report. If there is no previous inspection data, set the value to 0.
    '''
    X.insp_date = pd.to_datetime(X.insp_date).dt.date
    X.sort_values(by=['yelp_id', 'insp_date'], ascending=[True, True], inplace=True)
    violations = ['crit_viol', 'non_crit_viol', 'crit_viol_cos', 'crit_viol_rpt', 'non_crit_viol_cos', 'non_crit_viol_rpt', 'crit_viol_tbr', 'non_crit_viol_tbr']

    for x in violations:
        col_name = 'prev_' + x
        X[col_name] = X.groupby('yelp_id', sort=False)[x].shift(-1)
        X[col_name] = X[col_name].fillna(0)

    return X

def create_target(X):
    '''
    Create the target values for classification.
    0 = no critical violations
    1 = at least 1 critical violation
    '''

    X['target'] = [0 if x == 0 else 1 for x in X.crit_viol]
    return X

def yelp_dummies(X):
    '''
    Turn the yelp categories feature into dummy variables for analysis.
    '''
    X.yelp_cat.replace(regex=True, to_replace='\[*\]*', value='', inplace=True)
    X.yelp_cat = X.yelp_cat.apply(lst)
    category_dummies = pd.get_dummies(X.yelp_cat.apply(pd.Series).stack()).sum(level=0)
    result = X.join(category_dummies)
    return result

def lst(x):
    '''
    Splits the list of categories and takes every other one, based on how yelp returns this data from the API. Future build-- move to the yelp api gathering/ cleaning process.
    '''
    return x.split(", ")[1::2]

def create_datasets(df, dummies='no'):

    #add the prev violation info
    df = prev_viol(df)
    #create a dataframe with the time_diff column
    df = time_diff_computation(df)
    #convert yelp data to dummies
    df = yelp_dummies(df)
    #create the target value
    df = create_target(df)
    #use data from 3/16-5/16
    df.insp_date = pd.to_datetime(df.insp_date).dt.date
    df = df[(df.orig_insp_date >= datetime.date(year=2016,month=3,day=1)) & (df.insp_date <= datetime.date(year=2016,month=5,day=31))]
    #change temp to 3day avg high for 3/16 (60.52)
    df.avg_high_temp = 60.52
    #write to csv
    df.to_csv('metrics.csv', index=False)
    return df

if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    os.system('clear')

    df = pd.read_csv('multiple_sources_dataset.csv')
    data = create_datasets(df)
