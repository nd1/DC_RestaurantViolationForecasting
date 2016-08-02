'''
Score different estimators in order to determine which one to hypertune.

Nicole Donnelly 20160731
'''

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import os
import pickle
import time
import warnings

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cross_validation import StratifiedKFold, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.grid_search import GridSearchCV
from sklearn.linear_model import ElasticNetCV, LogisticRegressionCV, LogisticRegression, SGDClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from sklearn.svm import LinearSVC

BASE = os.path.abspath(os.path.join('.', 'data'))
OUTPATH = os.path.abspath(os.path.join('.', 'output'))
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

def generate_dataset(df, y=None):
    """
    Function to finalize dataset. Removes categorical features and drops numeric ones that will not be used.
    """
    df = df.loc[:, df.dtypes != object]
    for col in df.columns:
        if len(df[col].unique()) == df.shape[0]:
            df.drop(col, axis=1, inplace=True)
    df.drop(['doh_id', 'lat', 'lon', 'license_number'], axis=1, inplace=True)
    return df

def model_selection(train_data, feature_model, model_estimator, fse_label, model_label):

    """
    Test various combinations of estimators for feature selection and modeling.
    The pipeline generates the dataset, encodes columns based on the information they contain, then uses the encoded results for feature selection. Finally,
    the selected features are sent to the estimator model for scoring.
    """
    start  = time.time()

    X = train_data.ix[:, 0:-1]
    y = train_data.ix[:,-1]

    model = Pipeline([
        ('columns', FunctionTransformer(generate_dataset, validate=False)),
        ('features', FeatureUnion([
                    ('standard', Pipeline([
                                ('select', ColumnSelector(STANDARD_FEATURES)),
                                ('preprocessing', StandardScaler())])),
                    ('labels', Pipeline([
                                ('select', ColumnSelector(LABELS)),
                                ('preprocessing', OneHotEncoder())])),
                    ('no_scale', ColumnSelector(NO_SCALE))
                    ])),
        ('feature_selection', SelectFromModel(feature_model)),
        ('estimator', model_estimator)])

    """
    Train and test the model using StratifiedKFold cross validation. Compile the scores for each iteration of the model.
    """
    scores = {'accuracy':[], 'auc':[], 'f1':[], 'precision':[], 'recall':[]}
    for train, test in StratifiedKFold(y, n_folds=12, shuffle=True):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10)

        model.fit(X_train, y_train)
        expected  = y_test
        predicted = model.predict(X_test)

        scores['accuracy'].append(accuracy_score(expected, predicted))
        scores['f1'].append(f1_score(expected, predicted, average='binary'))
        scores['precision'].append(precision_score(expected, predicted, average='binary'))
        scores['recall'].append(recall_score(expected, predicted, average='binary'))

        """
        AUC cannot be computed if only 1 class is represented in the data. When that happens, record an AUC score of 0.
        """
        try:
            scores['auc'].append(roc_auc_score(expected, predicted))
        except:
            scores['auc'].append(0)

    """
    Print the modeling details and the mean score.
    """
    print "\nBuild and Validation of took {:0.3f} seconds\n".format(time.time()-start)
    print "Feature Selection Estimator: {}\n".format(fse_label)
    print "Estimator Model: {}\n".format(model_label)
    print "Validation scores are as follows:\n"
    print pd.DataFrame(scores).mean()

    """
    Create a data frame with the mean score and estimator details.
    """
    data = pd.DataFrame(scores).mean()
    data['SelectFromModel'] =  fse_label
    data['Estimator']  = model_label

    """
    Write official estimator to disk
    """
    estimator = model
    estimator.fit(X,y)

    outpath = os.path.join(OUTPATH + "/", fse_label.lower().replace(" ", "-") + "_" + model_label.lower().replace(" ", "-") + ".pickle")
    with open(outpath, 'w') as f:
        pickle.dump(estimator, f)

    print "\nFitted model written to:\n{}".format(os.path.abspath(outpath))

    return data

if __name__ == '__main__':

    """
    Use the training data to evalauate models. Compile a dataframe of the results to write to CSV for review.
    """

    warnings.filterwarnings("ignore")
    os.system('clear')

    TRAIN = os.path.join(BASE, 'train.csv')
    train_data = pd.read_csv(TRAIN)

    evaluation = pd.DataFrame()
    evaluation = evaluation.append(pd.DataFrame(model_selection(train_data, ElasticNetCV(), LogisticRegression(), "ElasticNetCV", "LogisticRegression")).T)
    evaluation = evaluation.append(pd.DataFrame(model_selection(train_data, LogisticRegressionCV(), LogisticRegression(), "LogisticRegressionCV", "LogisticRegression")).T)
    evaluation = evaluation.append(pd.DataFrame(model_selection(train_data, LinearSVC(), LogisticRegression(), "LinearSVC", "LogisticRegression")).T)
    evaluation = evaluation.append(pd.DataFrame(model_selection(train_data, SGDClassifier(), LogisticRegression(), "SGDClassifier", "LogisticRegression")).T)
    evaluation = evaluation.append(pd.DataFrame(model_selection(train_data, RandomForestClassifier(), LogisticRegression(), "RandomForestClassifier", "LogisticRegression")).T)

    evaluation = evaluation.append(pd.DataFrame(model_selection(train_data, ElasticNetCV(), KNeighborsClassifier(), "ElasticNetCV", "KNeighborsClassifier")).T)
    evaluation = evaluation.append(pd.DataFrame(model_selection(train_data, LogisticRegressionCV(), KNeighborsClassifier(), "LogisticRegressionCV", "KNeighborsClassifier")).T)
    evaluation = evaluation.append(pd.DataFrame(model_selection(train_data, LinearSVC(), KNeighborsClassifier(), "LinearSVC", "KNeighborsClassifier")).T)
    evaluation = evaluation.append(pd.DataFrame(model_selection(train_data, SGDClassifier(), KNeighborsClassifier(), "SGDClassifier", "KNeighborsClassifier")).T)
    evaluation = evaluation.append(pd.DataFrame(model_selection(train_data, RandomForestClassifier(), KNeighborsClassifier(), "RandomForestClassifier", "KNeighborsClassifier")).T)

    evaluation = evaluation.append(pd.DataFrame(model_selection(train_data, ElasticNetCV(), RandomForestClassifier(), "ElasticNetCV", "RandomForestClassifier")).T)
    evaluation = evaluation.append(pd.DataFrame(model_selection(train_data, LogisticRegressionCV(), RandomForestClassifier(), "LogisticRegressionCV", "RandomForestClassifier")).T)
    evaluation = evaluation.append(pd.DataFrame(model_selection(train_data, LinearSVC(), RandomForestClassifier(), "LinearSVC", "RandomForestClassifier")).T)
    evaluation = evaluation.append(pd.DataFrame(model_selection(train_data, SGDClassifier(), RandomForestClassifier(), "SGDClassifier", "RandomForestClassifier")).T)
    evaluation = evaluation.append(pd.DataFrame(model_selection(train_data, RandomForestClassifier(), RandomForestClassifier(), "RandomForestClassifier", "RandomForestClassifier")).T)

    evaluation = evaluation.append(pd.DataFrame(model_selection(train_data, ElasticNetCV(), SGDClassifier(), "ElasticNetCV", "SGDClassifier")).T)
    evaluation = evaluation.append(pd.DataFrame(model_selection(train_data, LogisticRegressionCV(), SGDClassifier(), "LogisticRegressionCV", "SGDClassifier")).T)
    evaluation = evaluation.append(pd.DataFrame(model_selection(train_data, LinearSVC(), SGDClassifier(), "LinearSVC", "SGDClassifier")).T)
    evaluation = evaluation.append(pd.DataFrame(model_selection(train_data, SGDClassifier(), SGDClassifier(), "SGDClassifier", "SGDClassifier")).T)
    evaluation = evaluation.append(pd.DataFrame(model_selection(train_data, RandomForestClassifier(), SGDClassifier(), "RandomForestClassifier", "SGDClassifier")).T)

    evaluation = evaluation.append(pd.DataFrame(model_selection(train_data, ElasticNetCV(), LinearSVC(), "ElasticNetCV", "LinearSVC")).T)
    evaluation = evaluation.append(pd.DataFrame(model_selection(train_data, LogisticRegressionCV(), LinearSVC(), "LogisticRegressionCV", "LinearSVC")).T)
    evaluation = evaluation.append(pd.DataFrame(model_selection(train_data, LinearSVC(), LinearSVC(), "LinearSVC", "LinearSVC")).T)
    evaluation = evaluation.append(pd.DataFrame(model_selection(train_data, SGDClassifier(), LinearSVC(), "SGDClassifier", "LinearSVC")).T)
    evaluation = evaluation.append(pd.DataFrame(model_selection(train_data, RandomForestClassifier(), LinearSVC(), "RandomForestClassifier", "LinearSVC")).T)

    outpath = os.path.join(OUTPATH + "/" + "model_comparison.csv")
    evaluation.to_csv(outpath, index=False)
