import pandas as pd
from xgboost import XGBClassifier
from scipy.sparse import hstack
from sklearn.preprocessing import LabelEncoder
import os
from os import path, makedirs
from azureml.logging import get_azureml_logger
from sklearn.model_selection import GridSearchCV
import feature_engineering as fe
from load_data import load_data

# load data
app_events, app_labels, events, gender_age_train, gender_age_test, label_categories, brand_model = load_data()

# initialize logger
run_logger = get_azureml_logger()

# default temporary library of joblib is too small, change it
os.environ["JOBLIB_TEMP_FOLDER"] = "/tmp"

#################################################################
# Feature engineering
#################################################################  

# Create one-hot encoding of brand and model
train_brand, test_brand, train_model, test_model = fe.one_hot_brand_model(brand_model, gender_age_train, gender_age_test)

# Create weekday and hour features (represented using one-hot encoding)
train_weekday, train_hour, train_weekday_hour, test_weekday, test_hour, test_weekday_hour = fe.weekday_hour_features(events, gender_age_train, gender_age_test)                                                                       

# Create one-hot encoding of apps and labels
train_app, train_label, test_app, test_label, device_id_label = fe.one_hot_app_labels(app_events, app_labels, events, gender_age_train, gender_age_test)

# Create text category features
train_category, test_category, device_id_label_category = fe.text_category_features(device_id_label, label_categories, gender_age_train, gender_age_test)

# Create category one-hot encoding features
train_category_word, test_category_word = fe.one_hot_category(device_id_label_category, gender_age_train, gender_age_test)

#############################################################
# Create training and test sets
#############################################################  

X_train = hstack((train_brand, train_model, train_app, train_label, train_category, train_weekday, train_hour, 
                  train_weekday_hour, train_category_word), format='csr')
X_test = hstack((test_brand, test_model, test_app, test_label, test_category, test_weekday, test_hour, 
                 test_weekday_hour, test_category_word), format='csr')

targetencoder = LabelEncoder().fit(gender_age_train['group'])
y_train = targetencoder.transform(gender_age_train['group'])

######################################################
# Training
#######################################################

tuned_parameters = [{'n_estimators': [300,400], 'max_depth': [3,4], 'objective': ['multi:softprob'], 
                     'reg_alpha': [1], 'reg_lambda': [1], 'colsample_bytree': [1],
                     'learning_rate': [0.1], 'colsample_bylevel': [0.1], 'subsample': [0.5]}]                                                               

clf = XGBClassifier(seed=0)
metric = 'neg_log_loss'

clf_cv = GridSearchCV(clf, tuned_parameters, scoring=metric, cv=5, n_jobs=8, verbose=3)
model = clf_cv.fit(X_train,y_train)

run_logger.log(metric, float(clf_cv.best_score_))
for key in clf_cv.best_params_.keys():
    run_logger.log(key, clf_cv.best_params_[key])

if not path.exists('./outputs'):
    makedirs('./outputs')
outfile = open('./outputs/sweeping_results.txt','w')

print("metric = ", metric, file=outfile)
for i in range(len(model.grid_scores_)):
    print(model.grid_scores_[i], file=outfile)
outfile.close()

