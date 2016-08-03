'''
Score the model with out of sample data.
Incorporates code from Jonathan Boyle for plotting/ assement.

Nicole Donnelly 20160802
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
from sklearn.metrics import accuracy_score, auc, classification_report, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score, roc_curve
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from sklearn.svm import LinearSVC

BASE = os.path.abspath(os.path.join('.', 'data'))
OUTPATH = os.path.abspath(os.path.join('.', 'testing'))
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

def class_rpt(Y_test, Y_pred):

    target_names = ['No_crit_viol', 'Crit_viol']
    clr = classification_report(Y_test, Y_pred, target_names=target_names)
    print clr

def compute_cm(Y_test, Y_pred, model_label):
    cm = confusion_matrix(Y_test, Y_pred)
    np.set_printoptions(precision=2)
    print "\nConfusion matrix, without normalization\n"
    print cm
    plot_confusion_matrix(cm, model_label)

    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    print '\nNormalized Confusion Matrix\n'
    print(cm_normalized)
    plot_confusion_matrix(cm_normalized, model_label, title='Normalized Confusion Matrix')

def plot_confusion_matrix(cm, model_label, title='Confusion Matrix', cmap=plt.cm.Blues):

    target_names = ['No_crit_viol', 'Crit_viol']
    output_file = os.path.join(OUTPATH + '/' + model_label + title.lower().strip() + '.png')
    plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, size = 15)
    plt.colorbar()
    tick_marks = np.arange(len(target_names))
    plt.xticks(tick_marks, target_names, rotation=0, size = 12)
    plt.yticks(tick_marks, target_names, rotation=90, size = 12)
    plt.tight_layout()
    plt.ylabel('True Label', size = 15)
    plt.xlabel('Predicted Label', size = 15)
    plt.savefig(output_file)
    plt.close
    print "\nSaved confusion matrix plot.\n"

def roc_curve_single_class(Y_test, Y_score, model_label):

    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    fpr[1], tpr[1], _ = roc_curve(Y_test, Y_score)
    roc_auc[1] = auc(fpr[1], tpr[1])

    output_file = os.path.join(OUTPATH + '/' + model_label + 'ROC.png')

    plt.figure()
    plt.plot(fpr[1], tpr[1], label='ROC curve (area = %0.2f)' % roc_auc[1])
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', size = 15)
    plt.ylabel('True Positive Rate', size = 15)
    plt.xticks(size = 12)
    plt.yticks(size = 12)
    plt.title('Receiver Operating Characteristic (ROC) for Violation Case', size = 15)
    plt.legend(loc="lower right")
    plt.savefig(output_file)
    plt.close
    print "\nSaved ROC curve.\n"

def model_scoring(data, pickled_estimator, label):

    X = data.ix[:, 0:-1]
    y = data.ix[:,-1]
    locations = data[['yelp_id']]

    model = load_model(pickled_estimator)

    y_pred = model.predict(X)
    y_pp = model.decision_function(X)

    print "Accuracy Score: ", round(accuracy_score(y, y_pred), 3)
    print "F1 Score: ", round(f1_score(y, y_pred), 3)
    print "Precision Score: ", round(precision_score(y, y_pred), 3)
    print "Recall Score: ", round(recall_score(y, y_pred), 3)
    print "AUC: ", round(roc_auc_score(y, y_pred), 3)

    print "\nClassification Report:\n"
    class_rpt(y, y_pred)

    compute_cm(y, y_pred, label)

    roc_curve_single_class(y, y_pred, label)

    #write output file
    output_file = os.path.join(OUTPATH + '/' + label + '_model_predictions.csv')
    results_df = locations
    results_df = results_df.join(y)
    results_df['predicted_result'] = pd.Series(y_pred)
    results_df['decision_function'] = pd.Series(y_pp)
    results_df.to_csv(output_file, index=False)

if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    os.system('clear')

    """
    Load the out of sample dataset and model it with different models pickled in the evaluation process.
    """
    OOS = os.path.join(BASE, 'oos.csv')
    oos_data = pd.read_csv(OOS)

    model_scoring(oos_data, 'output/sgdclassifier_linearsvc.pickle', "SGD-LinearSVC")

    model_scoring(oos_data, 'output/logisticregressioncv_linearsvc.pickle', "LogReg-LinearSVC")
