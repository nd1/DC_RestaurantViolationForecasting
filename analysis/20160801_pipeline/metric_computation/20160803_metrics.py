'''
Compute metrics data for evaluation.

Nicole Donnelly 20160803
'''

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import os
import pickle
import warnings

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from sklearn.svm import LinearSVC

BASE = os.path.abspath(os.path.join('.', 'data'))
OUTPATH = os.path.abspath(os.path.join('.', 'scoring'))
LABELS=['insp_badge']
STANDARD_FEATURES=['crit_viol', 'non_crit_viol', 'crit_viol_cos', 'crit_viol_rpt', 'non_crit_viol_cos', 'non_crit_viol_rpt', 'crit_viol_tbr', 'non_crit_viol_tbr', 'yelp_rating', 'yelp_reviews', 'risk', 'crime_count', '311_count', 'construction_count', 'avg_high_temp', 'time_diff', 'prev_crit_viol']
NO_SCALE=['Burgers', 'Convenience Stores', 'Sandwiches', 'Wine & Spirits', 'adultentertainment', 'afghani', 'african', 'apartments', 'asianfusion', 'bagels', 'bakeries', 'bangladeshi', 'bars', 'bbq', 'beerbar', 'beergardens', 'belgian', 'brasseries', 'breakfast_brunch', 'breweries', 'british', 'buffets', 'burgers', 'burmese', 'cafes', 'cafeteria', 'cajun', 'catering', 'cheesesteaks', 'chicken_wings', 'chinese',  'chocolate', 'churches', 'cocktailbars', 'coffee', 'coffeeroasteries', 'comfortfood', 'cookingschools', 'creperies', 'cuban', 'cupcakes', 'danceclubs', 'delis', 'desserts', 'diners', 'discountstore', 'divebars', 'donuts', 'drugstores', 'ethiopian', 'ethnicmarkets', 'falafel', 'foodtrucks', 'french', 'gastropubs', 'gelato', 'german', 'gluten_free', 'golf', 'gourmet', 'greek', 'grocery', 'gyms', 'halal', 'healthtrainers', 'hookah_bars', 'hotdog', 'hotdogs', 'hotels', 'icecream', 'indpak', 'irish', 'irish_pubs', 'italian', 'japanese', 'jazzandblues', 'juicebars', 'korean', 'landmarks', 'latin', 'lawyers', 'lebanese', 'libraries', 'lounges', 'mediterranean', 'mexican', 'mideastern', 'mini_golf', 'modern_european', 'musicvenues',  'newamerican', 'nonprofit', 'pakistani', 'peruvian', 'pianobars', 'pizza', 'publicservicesgovt', 'pubs', 'puertorican', 'restaurants', 'salad', 'salvadoran', 'sandwiches', 'seafood', 'social_clubs', 'soulfood', 'soup', 'southern', 'spanish', 'sports_clubs', 'sportsbars', 'steak', 'sushi', 'tapas', 'tapasmallplates', 'tea', 'tex-mex', 'thai', 'tobaccoshops', 'tradamerican', 'turkish', 'vegetarian', 'venues', 'vietnamese', 'wholesale_stores', 'wine_bars']

class ColumnSelector(BaseEstimator, TransformerMixin):

    def __init__(self, column_names):
        self.column_names = column_names

    def fit(self, X, y=None):
        return self

    def transform(self, df):
        """
        Expects a dataframe that contains the specified columns. Returns those columns.
        """
        return df[self.column_names]

def load_model(path):
    """
    Load the pickled model.
    """
    with open(path, 'rb') as f:
        return pickle.load(f)



def metrics_eval(data, pickled_estimator, label):

    X = data.ix[:, 0:-1]

    model = load_model(pickled_estimator)
    y_pred = model.predict(X)
    y_pp = model.decision_function(X)

    #write output file
    output_file = os.path.join(OUTPATH + '/' + label + '_model_eval.csv')
    results_df =  data
    results_df['predicted_result'] = pd.Series(y_pred)
    results_df['decision_function'] = pd.Series(y_pp)
    results_df.to_csv(output_file, index=False)

if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    os.system('clear')

    """
    Load the out of metric predictionn dataset and model it.
    """
    METRICS = os.path.join(BASE, 'metrics.csv')
    metrics_data = pd.read_csv(METRICS)

    metrics_eval(metrics_data, 'output/linearsvc.pickle', "LinearSVC")
