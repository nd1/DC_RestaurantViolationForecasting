'''
Model the data with KNeighborsClassifier

Incorporates code from Jonathan Boyle for plotting/ assement.

Nicole Donnelly 20160716
'''

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import warnings

from sklearn.ensemble import RandomForestClassifier
from sklearn.grid_search import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc, classification_report, confusion_matrix, accuracy_score, f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn_pandas import DataFrameMapper


def feature_selection(X, y, dataset):
    #use random forest to select features
    rfc = RandomForestClassifier()
    #GridSearch random forest for optimum parameters.
    PARAMETERS = {'n_estimators':[10,20,40,60,80,100,200,400,600],
                  'criterion':['gini', 'entropy'],
                  'class_weight':['balanced','balanced_subsample'],
                  'max_features':['auto',0.33],
                  'min_samples_leaf':[1,2,5,10,20]}

    print "\nStarting RFC GridSearchCV\n"
    rfc_estimator, rfc_score, rfc_params = grid_search(X, y, rfc, PARAMETERS)

    print "\nBest Estimator for %s\n" % dataset
    print rfc_estimator
    print "\nBest Score\n"
    print rfc_score

    #use parameters from grid search to run model and print feature importances
    rfc_ft_selection = RandomForestClassifier(max_features=rfc_params['max_features'], n_estimators=rfc_params['n_estimators'], min_samples_leaf=rfc_params['min_samples_leaf'], criterion=rfc_params['criterion'], class_weight=rfc_params['class_weight'])

    rfc_ft_selection.fit(X,y)
    importances = rfc_ft_selection.feature_importances_
    std = np.std([tree.feature_importances_ for tree in rfc_ft_selection.estimators_],axis=0)
    indices = np.argsort(importances)[::-1]

    print "\nRandom Forest feature importances\n"

    col_list = []
    for f in range(X.shape[1]):
        print("%d. %s feature %d (%f)" % (f + 1, X.columns[indices[f]], indices[f], importances[indices[f]]))
        if importances[indices[f]] >= .01:
            col_list.append(X.columns[indices[f]])
    print col_list
    return col_list

def grid_search(X, y, model, gs_params):

    grid = GridSearchCV(model, gs_params, verbose=True, n_jobs=-1, cv=12)
    grid.fit(X, y)
    estimator = grid.best_estimator_
    score = grid.best_score_
    model_params = grid.best_params_
    return estimator, score, model_params

def class_rpt(Y_test, Y_pred):

    target_names = ['Less_crit_viol', 'More_crit_viol']
    clr = classification_report(Y_test, Y_pred, target_names=target_names)
    print clr

def plot_confusion_matrix(cm, dataset, title='Confusion Matrix', cmap=plt.cm.Blues):

    target_names = ['Less_crit_viol', 'More_crit_viol']
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
    plt.savefig(dataset + title.lower().strip() + '.png')
    plt.close
    print "\nSaved confusion matrix plot.\n"

def compute_cm(Y_test, Y_pred, dataset):
    cm = confusion_matrix(Y_test, Y_pred)
    np.set_printoptions(precision=2)
    print "\nConfusion matrix, without normalization\n"
    print cm
    plot_confusion_matrix(cm, dataset)

    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    print '\nNormalized Confusion Matrix\n'
    print(cm_normalized)
    plot_confusion_matrix(cm_normalized, dataset, title='Normalized Confusion Matrix')

def roc_curve_single_class(Y_test, Y_score, dataset):

    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    fpr[1], tpr[1], _ = roc_curve(Y_test, Y_score)
    roc_auc[1] = auc(fpr[1], tpr[1])

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
    plt.savefig(dataset + 'ROC.png')
    plt.close
    print "\nSaved ROC curve.\n"

def scale_X(X, dataset):
    if dataset == 'noYelp':
        X_scaled = pd.DataFrame(StandardScaler().fit_transform(X), columns=X.columns)
    else:
        #use sklearn pandas data mapper to scale only non binary columns
        mapper = DataFrameMapper([(['yelp_rating'], StandardScaler()), (['yelp_reviews'], StandardScaler()), (['risk'], StandardScaler()), (['insp_badge'], StandardScaler()), (['crime_count'], StandardScaler()), (['311_count'], StandardScaler()), (['construction_count'], StandardScaler()), (['avg_high_temp'], StandardScaler()), (['time_diff'], StandardScaler()), (['prev_crit_viol'], StandardScaler()), ('Burgers', None), ('Convenience Stores', None), ('Sandwiches', None), ('Wine & Spirits', None), ('adultentertainment', None), ('afghani', None), ('african', None), ('apartments', None), ('asianfusion', None), ('bagels', None), ('bakeries', None), ('bangladeshi', None), ('bars', None), ('bbq', None), ('beerbar', None), ('beergardens', None), ('belgian', None), ('brasseries', None), ('breakfast_brunch', None), ('breweries', None), ('british', None), ('buffets', None), ('burgers', None), ('burmese', None), ('cafes', None), ('cafeteria', None), ('cajun', None), ('catering', None), ('cheesesteaks', None), ('chicken_wings', None), ('chinese', None), ('chocolate', None), ('churches', None),('cocktailbars', None), ('coffee', None), ('coffeeroasteries', None), ('comfortfood', None), ('cookingschools', None), ('creperies', None), ('cuban', None), ('cupcakes', None), ('danceclubs', None), ('delis', None), ('desserts', None), ('diners', None), ('discountstore', None), ('divebars', None), ('donuts', None), ('drugstores', None), ('ethiopian', None), ('ethnicmarkets', None), ('falafel', None), ('foodtrucks', None), ('french', None), ('gastropubs', None), ('gelato', None), ('german', None), ('gluten_free', None), ('golf', None), ('gourmet', None), ('greek', None), ('grocery', None), ('gyms', None), ('halal', None), ('healthtrainers', None), ('hookah_bars', None),  ('hotdog', None), ('hotdogs', None), ('hotels', None), ('icecream', None), ('indpak', None), ('irish', None), ('irish_pubs', None), ('italian', None), ('japanese', None),  ('jazzandblues', None), ('juicebars', None), ('korean', None), ('landmarks', None),  ('latin', None), ('lawyers', None), ('lebanese', None), ('libraries', None), ('lounges', None), ('mediterranean', None), ('mexican', None), ('mideastern', None), ('mini_golf', None), ('modern_european', None), ('musicvenues', None), ('newamerican', None), ('nonprofit', None), ('pakistani', None), ('peruvian', None), ('pianobars', None), ('pizza', None),  ('publicservicesgovt', None), ('pubs', None), ('puertorican', None), ('restaurants', None),  ('salad', None), ('salvadoran', None), ('sandwiches', None), ('seafood', None),  ('social_clubs', None), ('soulfood', None), ('soup', None), ('southern', None),  ('spanish', None), ('sports_clubs', None), ('sportsbars', None), ('steak', None), ('sushi', None), ('tapas', None), ('tapasmallplates', None), ('tea', None),  ('tex-mex', None), ('thai', None), ('tobaccoshops', None), ('tradamerican', None), ('turkish', None), ('vegetarian', None), ('venues', None), ('vietnamese', None), ('wholesale_stores', None), ('wine_bars', None)])

        X_scaled = pd.DataFrame(mapper.fit_transform(X.copy()), columns=X.columns)

    print "\n data scaled\n"
    return X_scaled

def compute_results(clf, oos, selected_features, dataset):

    print "\nComputing results \n"
    locations = oos[['yelp_id']]
    X = oos.drop(['yelp_id', 'target'], axis=1)

    #scale the data
    #future upgrade -- use the mapper with the reduced set of features
    X_scaled = scale_X(X, dataset)
    X_new = X_scaled[selected_features]
    y = oos.ix[:,11]
    y_pred = clf.predict(X_new)
    y_pp = clf.predict_proba(X_new)

    print "Accuracy Score: ", round(accuracy_score(y, y_pred), 3)
    print "F1 Score: ", round(f1_score(y, y_pred), 3)

    print "\nClassification Report:\n"
    class_rpt(y, y_pred)

    compute_cm(y, y_pred, dataset)

    roc_curve_single_class(y, y_pred, dataset)

    #write output file
    output_file = dataset + '_model_predictions.csv'
    results_df = locations
    results_df = results_df.join(y)
    results_df['predicted_result'] = pd.Series(y_pred)
    results_df['predicted_proba_0'] = pd.Series(y_pp[:,0])
    results_df['predicted_proba_1'] = pd.Series(y_pp[:,1])
    results_df.to_csv(output_file, index=False)

def knn_model(df, oos, dataset):

    if 'yelp_id' in df.columns:
        df.drop('yelp_id', axis=1, inplace=True)

    X = df.drop('target', axis=1)
    y = df.ix[:,10]

    selected_features = feature_selection(X, y, dataset)
    #scale the data
    X_scaled = scale_X(X, dataset)
    print X_scaled.head()
    X_new = X_scaled[selected_features]

    #grid search knn
    knn = KNeighborsClassifier()
    knn_params = {'n_neighbors':[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20],
                  'weights':['uniform','distance'],
                  'algorithm':['ball_tree','kd_tree','brute','auto']}
    print "\nStarting KNeighborsClassifier GridSearchCV\n"
    knn_estimator, knn_score, knn_params = grid_search(X_new, y, knn, knn_params)

    print "\nBest Estimator for %s\n" % dataset
    print knn_estimator
    print "\nBest Score\n"
    print knn_score

    clf = KNeighborsClassifier(algorithm=knn_params['algorithm'], n_neighbors=knn_params['n_neighbors'], weights=knn_params['weights'])

    #train data on 2013-2015, compute results on out of sample data
    clf_model = clf.fit(X_new, y)
    print "Score on scaled data %r\n" % clf_model.score(X_new, y)
    compute_results(clf_model, oos, selected_features, dataset)

if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    os.system('clear')
    print "Modeling: no categories\n"
    noYelp = pd.read_csv('df_noYelp_train.csv')
    noYelp_oos = pd.read_csv('df_noYelp_oos.csv')
    knn_model(noYelp, noYelp_oos, 'noYelp')

    print "Modeling: categories\n"
    Yelp = pd.read_csv('df_Yelp_train.csv')
    Yelp_oos = pd.read_csv('df_Yelp_oos.csv')
    knn_model(Yelp, Yelp_oos, 'Yelp')
