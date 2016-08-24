#auc and confusion matrix code
#combination of scikit-learn documentation and Joseph's lecture
#https://github.com/ga-students/DSI-DC-1/blob/b3c5032e28f143059cd595daa7689dd39c728305/week-04/3.4-model-evaluation/model-evaluation.py
#http://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html
#http://scikit-learn.org/stable/auto_examples/model_selection/plot_precision_recall.html
#http://scikit-learn.org/stable/auto_examples/feature_selection/plot_rfe_with_cross_validation.html
#http://scikit-learn.org/stable/auto_examples/model_selection/plot_validation_curve.html
#http://scikit-learn.org/stable/auto_examples/ensemble/plot_forest_importances.html#example-ensemble-plot-forest-importances-py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, classification_report, confusion_matrix
from sklearn.metrics import accuracy_score, f1_score
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression
#from sklearn.feature_selection import RFECV
from sklearn.grid_search import GridSearchCV
from sklearn.learning_curve import validation_curve, learning_curve
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import pickle
from sklearn.neighbors import KNeighborsClassifier

df = pd.read_csv('/home/jonathan/WinUbuntu/Python Scripts/Capstone-Nicole/modeling_data20160707/50_50/df_Yelp_train.csv')
df2 = pd.read_csv('/home/jonathan/WinUbuntu/Python Scripts/Capstone-Nicole/modeling_data20160707/50_50/df_Yelp_oos.csv')

X = df.iloc[:,0:138]
X.shape
X
X.columns
Y = df.iloc[:,138]
Y.shape
Y
X = X.drop('yelp_id', axis = 1)
#X = X.drop('Unnamed: 0', axis = 1)
#X = X.drop('crit_viol', axis = 1)
X
Y

#Run GridSearch on Random Forest Classifier as the feature selector. Then use selected
# features for Logistic Regression model, run GridSearch on Logistic Regression to optimize
# model, and produce confusion matrices. Also run LogReg model on 2016 data. 

#Initialize RandomForestClassifier estimator
rfc = RandomForestClassifier()


#GridSearch for optimum parameters.
PARAMETERS = {'n_estimators':[10,20,40,60,80,100,200,400,600], 
              'criterion':['gini', 'entropy'],
              'class_weight':['balanced','balanced_subsample'],
              'max_features':['auto',0.33],
              'min_samples_leaf':[1,2,5,10,20]}
rfc_gridcv = GridSearchCV(rfc, PARAMETERS, verbose=True, n_jobs=-1, cv = 7)
rfc_gridcv.fit(X, Y)

print rfc_gridcv.best_estimator_
print rfc_gridcv.best_score_


#Input optimum parameters into RFC model
rfc = RandomForestClassifier(criterion = 'gini', n_estimators = 400, 
                             class_weight = 'balanced_subsample', max_features = 0.33,
                             min_samples_leaf = 10)
rfc.fit(X, Y)


#Plot number of features vs. cross-validation scores
def plot_features_scores(rfc):
    plt.figure()
    plt.xlabel("Number of features selected", size = 15)
    plt.ylabel("Score", size = 15)
    plt.xticks(size = 12)
    plt.yticks(size = 12)
    plt.title('RFC Feature Selection', size = 15)
    plt.plot(range(1, len(rfc.feature_importances_) + 1), rfc.feature_importances_)
    plt.savefig('plot_features_scores_rf_asis')
    plt.show()

plot_features_scores(rfc)


#Print out features sorted by importance rank
importances = rfc.feature_importances_
std = np.std([tree.feature_importances_ for tree in rfc.estimators_],
             axis = 0)
indices = np.argsort(importances)[::-1]

print "Features sorted: "
rfc_feat_import_df = pd.DataFrame({'feature':X.columns,
                                     'importance':rfc.feature_importances_})
rfc_feat_import_df_sorted = rfc_feat_import_df.sort_values(by = 'importance'
                                        , ascending = False)
rfc_feat_import_df_sorted


#Plot the feature importances of the Random Forest
def bar_plot_with_error_bars_sorted_features(X, importances, std, indices):
    plt.figure()
    plt.title("Feature Importances", size = 15)
    plt.bar(range(X.shape[1]), importances[indices], color = "r",
            yerr = std[indices], align = "center")
    plt.xticks(range(X.shape[1]), indices, size = 12)
    plt.yticks(size = 12)
    plt.xlabel("Features", size = 15)
    plt.ylabel("Importance", size = 15)
    plt.xlim([-1, X.shape[1]])
    plt.savefig('plot_bar_chart_features_scores_rf_asis')
    plt.show()
    
bar_plot_with_error_bars_sorted_features(X, importances, std, indices)


#Input selected features into Logistic Regression. Keeping features with 0.01 importance
# or better
features = ['insp_badge', 'yelp_reviews', 'time_diff', 'avg_high_temp', 'crime_count',
            'construction_count', 'yelp_rating', 'risk', 'bakeries', 'burgers']

X_features = X
X_features = X_features[features]
X_features
          
          
#Plot learning curve to determine optimum test-train split size.
def plot_learning_curve(X, Y):
    train_sizes = np.array([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1])
    train_sizes, train_scores, test_scores = learning_curve(
                                                KNeighborsClassifier(),
                                                 X_features, Y, train_sizes = train_sizes,
                                                 cv = 7, n_jobs = -1, verbose = True)
    train_scores_mean = np.mean(train_scores, axis = 1)
    train_scores_std = np.std(train_scores, axis = 1)
    test_scores_mean = np.mean(test_scores, axis = 1)
    test_scores_std = np.mean(test_scores, axis = 1)
    
    plt.figure()    
    plt.grid()
    plt.title("Learning Curve with KNN", size = 15)
    plt.xlabel("Training Examples", size = 15)
    plt.ylabel("Score", size = 15)
    plt.xticks(size = 12)
    plt.yticks(size = 12)
    plt.ylim(0.4,0.8)
    plt.plot(train_sizes, train_scores_mean, label = "Training Score", marker = "o", color = "r")
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std, 
                     train_scores_mean + train_scores_std, alpha = 0.2, color = "r")
    plt.plot(train_sizes, test_scores_mean, label = "Cross-Validation Score", marker = "o", color = "g")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std, 
                     test_scores_mean + test_scores_std, alpha = 0.2, color = "g")
    plt.legend(loc = "best")
    plt.savefig('plot_learning_curve_rf_50_50')
    plt.show()

plot_learning_curve(X_features, Y)

X_train, X_test, Y_train, Y_test = train_test_split(X_features, Y, test_size=0.30)


#Scale X data, based on the mean of the training data.
#Dummy variable columns not to be scaled.
#Scaling the non-dummy variables though converts them into an array, making it hard to 
#merge scaled and unscaled data.
# Y data not scaled here since all values are 0 or 1.
X_train2 = X_train.iloc[:,0:8]
X_test2 = X_test.iloc[:,0:8]

X_train3 = X_train.iloc[:,8:12]
X_test3 = X_test.iloc[:,8:12]

std_scale = StandardScaler().fit(X_train2)
X_train_std = std_scale.transform(X_train2)
X_train_std = pd.concat([X_train_std,X_train3], axis = 1)
X_test_std = std_scale.transform(X_test2)
X_test_std = pd.concat([X_test_std,X_test3], axis = 1)


#Initialize LogisticRegression estimator
clf = KNeighborsClassifier()


#GridSearch for optimum parameters
PARAMETERS = {'n_neighbors':[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20],
              'weights':['uniform','distance'],
              'algorithm':['ball_tree','kd_tree','brute','auto']}
clf_gridcv = GridSearchCV(clf, PARAMETERS, verbose=True, n_jobs=-1, cv = 7)
clf_gridcv.fit(X_train, Y_train)

print clf_gridcv.best_estimator_
print clf_gridcv.best_score_


#Validation Curve. Similar to GridSearch with one variable, but also plots out test and
#train scores visually to determine whether the model is over- or under-fit.
def plot_validation_curve(clf, X_train, Y_train):
    param_range = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
    train_scores, test_scores = validation_curve(KNeighborsClassifier(algorithm = 'ball_tree',
                                                                    weights = 'uniform'),
                                                 X_train, Y_train, param_name = "n_neighbors",
                                                 param_range = param_range, cv = 7,
                                                 scoring = "accuracy", n_jobs = -1,
                                                 verbose = True)
    train_scores_mean = np.mean(train_scores, axis = 1)
    train_scores_std = np.std(train_scores, axis = 1)
    test_scores_mean = np.mean(test_scores, axis = 1)
    test_scores_std = np.mean(test_scores, axis = 1)
    
    plt.title("Validation Curve with KNN", size = 15)
    plt.xlabel("n_neighbors", size = 15)
    plt.ylabel("Score", size = 15)
    plt.xticks(size = 12)
    plt.yticks(size = 12)
    plt.ylim(0.5,1.0)
    plt.plot(param_range, train_scores_mean, label = "Training Score", color = "r")
    plt.fill_between(param_range, train_scores_mean - train_scores_std, 
                     train_scores_mean + train_scores_std, alpha = 0.2, color = "r")
    plt.plot(param_range, test_scores_mean, label = "Cross-Validation Score",
                 color = "g")
    plt.fill_between(param_range, test_scores_mean - test_scores_std, 
                     test_scores_mean + test_scores_std, alpha = 0.2, color = "g")
    plt.legend(loc = "best")
    plt.savefig('plot_validation_curve_rf_asis')
    plt.show()

plot_validation_curve(clf, X_train, Y_train)


#Manual input of best parameter values from GridSearch, need to automate this input.
clf = KNeighborsClassifier(n_neighbors = 5, algorithm= 'ball_tree', weights = 'distance')
clf.fit(X_train, Y_train)
#logreg.fit(X_train_std, Y_train)
model_save = pickle.dumps(clf)
Y_score_s = clf.fit(X_train, Y_train).predict_proba(X_test)[:,1]
#Y_score_s = logreg.fit(X_train_std, Y_train).decision_function(X_test_std)
Y_pred_s = clf.predict(X_test)
#Y_pred_s = logreg.predict(X_test_std)

#Print out data frame with coefficients and respective features
print "Features sorted: "
coef = logreg.coef_
coef

#Reshape coef to 1-dimensional shape
coef_trans = coef.reshape((10,))

logreg_feat_import_df = pd.DataFrame({'feature':X_features.columns,
                                     'coefficients':coef_trans})
logreg_feat_import_df_sorted = logreg_feat_import_df.sort_values(by = 'coefficients'
                                        , ascending = False)
logreg_feat_import_df_sorted

print "Accuracy Score: ", round(accuracy_score(Y_test, Y_pred_s), 3)
print "F1 Score: ", round(f1_score(Y_test, Y_pred_s), 3)


# Get the predicted probability vector
Y_pp = pd.DataFrame(logreg.predict_proba(X_test_std), columns=['class_0_pp','class_1_pp'])
print(Y_pp.iloc[0:18])


# Print classification report					 
target_names = ['Less_crit_viol', 'More_crit_viol']
clr = classification_report(Y_test, Y_pred_s, target_names=target_names)	
print clr
	
 
# Print/plot confusion matrix
cm = np.array(confusion_matrix(Y_test, Y_pred_s, labels=[0,1]))

confusion = pd.DataFrame(cm, index=['Less_crit_viol', 'More_crit_viol'],
                         columns=['predicted_less_crit_viol','predicted_crit_viol'])
confusion

def plot_confusion_matrix(cm, title='Confusion Matrix', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, size = 15)
    plt.colorbar()    
    tick_marks = np.arange(len(target_names))
    plt.xticks(tick_marks, target_names, rotation=0, size = 12)
    plt.yticks(tick_marks, target_names, rotation=90, size = 12)
    plt.tight_layout()
    plt.ylabel('True Label', size = 15)
    plt.xlabel('Predicted Label', size = 15)
    plt.savefig('plot_confusion_matrix_rf_asis')
    
# Compute confusion matrix
cm = confusion_matrix(Y_test, Y_pred_s)
np.set_printoptions(precision=2)
print('Confusion matrix, without normalization')
print(cm)
plt.figure()
plot_confusion_matrix(cm)


#Normalize the confusion matrix by row (i.e by the number of samples
# in each class)
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
print('Normalized Confusion Matrix')
print(cm_normalized)
plt.figure()
plot_confusion_matrix(cm_normalized, title='Normalized Confusion Matrix')
plt.savefig('plot_norm_confusion_matrix_rf_asis')
plt.show()


# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
fpr[1], tpr[1], _ = roc_curve(Y_test, Y_score_s)
roc_auc[1] = auc(fpr[1], tpr[1])
		
				 
# Plot of ROC curve for a specific class
def roc_curve_single_class(fpr, tpr, roc_auc):
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
    plt.savefig('plot_roc_curve_rf_asis')
    plt.show()
    
roc_curve_single_class(fpr, tpr, roc_auc)


#Scale 2016 test data from test.csv and input into already fit model.
#Test-train split not performed on 2016 data.
df2 = pd.read_csv('/home/jonathan/WinUbuntu/Python Scripts/Capstone-Nicole/modeling_data20160707/50_50/df_Yelp_oos.csv')

X2 = df2.iloc[:,0:138]
X2.shape
Y2 = df2.iloc[:,138]
Y2.shape
X2 = X2.drop('yelp_id', axis = 1)
#X = X.drop('Unnamed: 0', axis = 1)
#X = X.drop('crit_viol', axis = 1)
X2 = X2[features]
X2
Y2

#X2_scaled = StandardScaler().fit_transform(X2)

clf2 = pickle.loads(model_save)
Y2_score_s = clf2.predict_proba(X2)[:,1]
#Y2_score_s = logreg2.decision_function(X2_scaled)
Y2_pred_s = clf2.predict(X2)
#Y2_pred_s = logreg2.predict(X2_scaled)

print "Accuracy Score: ", round(accuracy_score(Y2, Y2_pred_s), 3)
print "F1 Score: ", round(f1_score(Y2, Y2_pred_s), 3)


# Print classification report					 
target_names = ['Less_crit_viol', 'More_crit_viol']
clr = classification_report(Y2, Y2_pred_s, target_names=target_names)	
print clr
	
 
# Print/plot confusion matrix
cm = np.array(confusion_matrix(Y2, Y2_pred_s, labels=[0,1]))

confusion = pd.DataFrame(cm, index=['Less_crit_viol', 'More_crit_viol'],
                         columns=['predicted_less_crit_viol','predicted_crit_viol'])
confusion


# Compute confusion matrix
cm = confusion_matrix(Y2, Y2_pred_s)
np.set_printoptions(precision=2)
print('Confusion matrix, without normalization')
print(cm)
plt.figure()
plt.savefig('plot_confusion_matrix_2016_rf_asis')
plot_confusion_matrix(cm)


#Normalize the confusion matrix by row (i.e by the number of samples
# in each class)
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
print('Normalized Confusion Matrix')
print(cm_normalized)
plt.figure()
plot_confusion_matrix(cm_normalized, title='Normalized Confusion Matrix')
plt.savefig('plot_confusion_matrix_2016_rf_asis')
plt.show()


# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
fpr[1], tpr[1], _ = roc_curve(Y2, Y2_score_s)
roc_auc[1] = auc(fpr[1], tpr[1])				 
   
roc_curve_single_class(fpr, tpr, roc_auc)