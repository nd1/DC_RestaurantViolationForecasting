'''
Using the dataset created from multiple sources, perform final feature extraction and wrangling to produce datasets for modeling.

20160731 Nicole Donnelly
'''

import datetime
import numpy as np
import pandas as pd

def time_diff_computation(X):
    '''
    Convert date values to date, sort the dataframe and use groupby to put the instances together for computation. The latest inspection will end up with NaT in the time_diff column. Compute the time difference between 6/14/16 and the last inspection.
    '''

    X.insp_date = pd.to_datetime(X.insp_date).dt.date
    X.sort_values(by=['yelp_id', 'insp_date'], ascending=[True, False], inplace=True)
    X['time_diff'] = X.groupby('yelp_id', sort=False)['insp_date'].diff()
    X.loc[X.time_diff.isnull(), 'time_diff'] = pd.to_timedelta(X.insp_date - datetime.date(year=2016,month=6,day=14))
    X.time_diff = [x.days for x in X.time_diff]
    return X

def prev_crit_viol(X):
    '''
    Add the count of previous critical violations on last inspection to the dataframe. If there is no previous inspection data, set the value to 0.
    '''
    X.insp_date = pd.to_datetime(X.insp_date).dt.date
    X.sort_values(by=['yelp_id', 'insp_date'], ascending=[True, False], inplace=True)
    X['prev_crit_viol'] = X.groupby('yelp_id', sort=False)['crit_viol'].shift(-1)
    X['prev_crit_viol'] = X['prev_crit_viol'].fillna(0)
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
    Turn the yelp categories feature into dummy variables for analysis. Move the targt variable to the end of the dataframe after joining dummies.
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

def data_split(df):
    '''
    Remove data from 2016 to be used as out of sample test data.
    '''
    train = df[df.insp_date < datetime.date(year=2016,month=1,day=1)]
    out_of_sample = df[df.insp_date >= datetime.date(year=2016,month=1,day=1)]
    return train, out_of_sample

def create_datasets(df, dummies='no'):

    #create a dataframe with the time_diff column
    df = time_diff_computation(df)
    #add the prev_crit_viol column
    df = prev_crit_viol(df)
    #convert yelp data to dummies
    df = yelp_dummies(df)
    #create the target value
    df = create_target(df)

    #split into training and 2016 data, write to csv
    train, oos = data_split(df)
    train.to_csv('train.csv', index=False)
    oos.to_csv('oos.csv', index=False)
    return train

if __name__ == '__main__':
    df = pd.read_csv('multiple_sources_dataset.csv')
    data = create_datasets(df)
