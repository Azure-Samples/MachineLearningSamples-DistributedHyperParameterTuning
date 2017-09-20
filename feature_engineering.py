from sklearn.preprocessing import LabelEncoder
from scipy.sparse import csr_matrix
import numpy as np
import pandas as pd

#############################################
# Create one-hot encoding of brand and model
#############################################

def one_hot_brand_model(brand_model, gender_age_train, gender_age_test):
    # convert strings to integers
    brandencoder = LabelEncoder().fit(brand_model['phone_brand'])
    brand_model['brand'] = brandencoder.transform(brand_model['phone_brand'])
    modelencoder = LabelEncoder().fit(brand_model['device_model'])
    brand_model['model'] = modelencoder.transform(brand_model['device_model'])

    # implicit join using indices of gender_age_train, gender_age_test and brand_model
    gender_age_train['brand'] = brand_model['brand']
    gender_age_test['brand'] = brand_model['brand']
    gender_age_train['model'] = brand_model['model']
    gender_age_test['model'] = brand_model['model']

    # get number of training and test examples
    n_train_examples = gender_age_train.shape[0]
    n_test_examples = gender_age_test.shape[0]

    gender_age_train['idtrain'] = np.arange(n_train_examples)
    gender_age_test['idtest'] = np.arange(n_test_examples)

    # create sparse matrices
    train_brand = csr_matrix((np.ones(n_train_examples),
                             (gender_age_train['idtrain'], gender_age_train['brand'])))
    test_brand = csr_matrix((np.ones(n_test_examples),
                            (gender_age_test['idtest'], gender_age_test['brand'])))
    train_model = csr_matrix((np.ones(n_train_examples),
                             (gender_age_train['idtrain'], gender_age_train['model'])))
    test_model = csr_matrix((np.ones(n_test_examples),
                            (gender_age_test['idtest'], gender_age_test['model'])))    

    return train_brand, test_brand, train_model, test_model                                                                          

##############################################
# Create weekday and hour features (represented using one-hot encoding)
##############################################

def weekday_hour_features(events, gender_age_train, gender_age_test):
    # compute weekday fraction per device_id
    events_weekday_count = events.groupby(['device_id','weekday']).size().reset_index()
    events_weekday_count.rename(columns={0: 'weekday_count'}, inplace=True)
    events_total = events.groupby(['device_id']).size().reset_index()
    events_total.rename(columns={0: 'total'}, inplace=True)
    events_weekday_count_total = events_weekday_count.merge(events_total,on='device_id')
    events_weekday_count_total['weekday_frac'] = events_weekday_count_total.apply(lambda x: x['weekday_count'] / x['total'], axis = 1)

    events_weekday_count_total['weekday_frac'] = 1

    # compute hour fraction per device_id
    events_hour_count = events.groupby(['device_id','hour']).size().reset_index()
    events_hour_count.rename(columns={0: 'hour_count'}, inplace=True)
    events_hour_count_total = events_hour_count.merge(events_total,on='device_id')
    events_hour_count_total['hour_frac'] = events_hour_count_total.apply(lambda x: x['hour_count'] / x['total'], axis = 1)

    events_hour_count_total['hour_frac'] = 1

    # inner join between device_app_count_total and gender_age_train by device_id, to get number of row of device_id 
    # in the original training set
    merged_train_weekday = events_weekday_count_total.merge(gender_age_train[['idtrain']], left_on='device_id', right_index=True)

    # inner join between device_app_count_total and gender_age_test by device_id, to get number of row of device_id 
    # in the original test set
    merged_test_weekday = events_weekday_count_total.merge(gender_age_test[['idtest']], left_on='device_id', right_index=True)

    # inner join between device_label_count_total and gender_age_train by device_id, to get number of row of device_id 
    # in the original training set
    merged_train_hour = events_hour_count_total.merge(gender_age_train[['idtrain']], left_on='device_id', right_index=True)

    # inner join between device_label_count_total and gender_age_test by device_id, to get number of row of device_id 
    # in the original test set
    merged_test_hour = events_hour_count_total.merge(gender_age_test[['idtest']], left_on='device_id', right_index=True)

    # get number of training and test examples
    n_train_examples = gender_age_train.shape[0]
    n_test_examples = gender_age_test.shape[0]

    # create sparse matrices with app features
    train_weekday = csr_matrix((merged_train_weekday['weekday_frac'],(merged_train_weekday['idtrain'],merged_train_weekday['weekday'])),
                           shape=(n_train_examples,7))
    test_weekday = csr_matrix((merged_test_weekday['weekday_frac'],(merged_test_weekday['idtest'],merged_test_weekday['weekday'])),
                           shape=(n_test_examples,7))

    # create sparse matrices with label features
    train_hour = csr_matrix((merged_train_hour['hour_frac'],(merged_train_hour['idtrain'],merged_train_hour['hour'])),
                           shape=(n_train_examples,24))
    test_hour = csr_matrix((merged_test_hour['hour_frac'],(merged_test_hour['idtest'],merged_test_hour['hour'])),
                           shape=(n_test_examples,24))

    # convert weekday_hour strings to integers
    weekdayhourencoder = LabelEncoder().fit(events['weekday_hour'])
    events['weekday_hour_id'] = weekdayhourencoder.transform(events['weekday_hour'])

    # compute weekday hour fraction per device_id
    events_weekday_hour_count = events.groupby(['device_id','weekday_hour_id']).size().reset_index()
    events_weekday_hour_count.rename(columns={0: 'weekday_hour_count'}, inplace=True)
    events_weekday_hour_count_total = events_weekday_hour_count.merge(events_total,on='device_id')
    events_weekday_hour_count_total['weekday_hour_frac'] = events_weekday_hour_count_total.apply(lambda x: x['weekday_hour_count'] / x['total'], axis = 1)

    events_weekday_hour_count_total['weekday_hour_frac'] = 1

    # inner join between device_label_co and gender_age_train by device_id, to get number of row of device_id 
    # in the original training set
    merged_train_weekday_hour = events_weekday_hour_count_total.merge(gender_age_train[['idtrain']], 
                                                                      left_on='device_id', right_index=True)

    # inner join between device_label_count_total and gender_age_test by device_id, to get number of row of device_id 
    # in the original test set
    merged_test_weekday_hour = events_weekday_hour_count_total.merge(gender_age_test[['idtest']], 
                                                                     left_on='device_id', right_index=True)

    # create sparse matrices with app features
    train_weekday_hour = csr_matrix((merged_train_weekday_hour['weekday_hour_frac'],
                                     (merged_train_weekday_hour['idtrain'],merged_train_weekday_hour['weekday_hour_id'])),
                                    shape=(n_train_examples,7*24))
    test_weekday_hour = csr_matrix((merged_test_weekday_hour['weekday_hour_frac'],
                                    (merged_test_weekday_hour['idtest'],merged_test_weekday_hour['weekday_hour_id'])),
                                   shape=(n_test_examples,7*24))

    return train_weekday, train_hour, train_weekday_hour, test_weekday, test_hour, test_weekday_hour 

####################################################
# Create one-hot encoding of apps and labels
####################################################

def one_hot_app_labels(app_events, app_labels, events, gender_age_train, gender_age_test):
    appencoder = LabelEncoder().fit(app_events['app_id'])
    app_events['app'] = appencoder.transform(app_events['app_id'])
    napps = len(appencoder.classes_)

    label_encoder = LabelEncoder().fit(app_labels['label_id'])
    app_labels['label'] = label_encoder.transform(app_labels['label_id'])
    nlabels = len(label_encoder.classes_)

    # left join app_events and events by event_id to map from device_id to app_id
    app_events['device_id'] = events['device_id']

    # left join app_events and labels by app_id to map from device_id to app_id
    device_id_app = app_events.reset_index()[['device_id','app','app_id']].drop_duplicates()
    merged = device_id_app.merge(app_labels, left_on='app_id', right_on='app_id')

    # generate unique triples (device_id,app_id,label)
    device_id_label = merged[['device_id','label','label_id']].drop_duplicates()
    device_id_label = device_id_label.set_index(['device_id'])

    # inner join between device_id_app and gender_age_train by device_id, to get number of row of device_id 
    # in the original training set
    merged_train_label = device_id_label.merge(gender_age_train[['idtrain']], left_index=True, right_index=True)
    device_id_app = device_id_app.set_index(['device_id'])
    merged_train = device_id_app.merge(gender_age_train[['idtrain']], left_index=True, right_index=True)

    # inner join between device_id_app and gender_age_test by device_id, to get number of row of device_id 
    # in the original test set
    merged_test_label = device_id_label.merge(gender_age_test[['idtest']], left_index=True, right_index=True)
    merged_test = device_id_app.merge(gender_age_test[['idtest']], left_index=True, right_index=True)

    # get number of training and test examples
    n_train_examples = gender_age_train.shape[0]
    n_test_examples = gender_age_test.shape[0]

    # create sparse matrices with app features
    train_app = csr_matrix((np.ones(merged_train.shape[0]),(merged_train['idtrain'],merged_train['app'])),
                           shape=(n_train_examples,napps))
    test_app = csr_matrix((np.ones(merged_test.shape[0]),(merged_test['idtest'],merged_test['app'])),
                           shape=(n_test_examples,napps))

    # create sparse matrices with label features
    train_label = csr_matrix((np.ones(merged_train_label.shape[0]),(merged_train_label['idtrain'],merged_train_label['label'])),
                           shape=(n_train_examples,nlabels))
    test_label = csr_matrix((np.ones(merged_test_label.shape[0]),(merged_test_label['idtest'],merged_test_label['label'])),
                           shape=(n_test_examples,nlabels))

    return train_app, train_label, test_app, test_label, device_id_label

###################################################
# Create text category features
###################################################  

def text_category_features(device_id_label, label_categories, gender_age_train, gender_age_test):
    device_id_label_category = device_id_label.reset_index().merge(label_categories, 
                                                                   left_on='label_id',right_on='label_id',how='left')

    categoryencoder = LabelEncoder().fit(device_id_label_category['category'])
    device_id_label_category['category_id'] = categoryencoder.transform(device_id_label_category['category'])
    ncategories = len(categoryencoder.classes_)
    device_id_label_category.set_index(['device_id'],inplace=True)

    # inner join between device_categories_word and gender_age_train by device_id, to get number of row of device_id 
    # in the original training set
    merged_train = device_id_label_category.merge(gender_age_train[['idtrain']], left_index=True, right_index=True)
    merged_train.drop_duplicates(['idtrain','category_id'],inplace=True)

    # inner join between device_categories_word and gender_age_test by device_id, to get number of row of device_id 
    # in the original test set
    merged_test = device_id_label_category.merge(gender_age_test[['idtest']], left_index=True, right_index=True)
    merged_test.drop_duplicates(['idtest','category_id'],inplace=True)

    # get number of training and test examples
    n_train_examples = gender_age_train.shape[0]
    n_test_examples = gender_age_test.shape[0]

    # create sparse matrices with app features
    train_category = csr_matrix((np.ones(merged_train.shape[0]),(merged_train['idtrain'],merged_train['category_id'])),
                           shape=(n_train_examples,ncategories))
    test_category = csr_matrix((np.ones(merged_test.shape[0]),(merged_test['idtest'],merged_test['category_id'])),
                           shape=(n_test_examples,ncategories))

    return train_category, test_category, device_id_label_category

#######################################################
# Create category one-hot encoding features
######################################################  

def one_hot_category(device_id_label_category, gender_age_train, gender_age_test):

    def get_words(x):
        all_words = ' '.join(x)
        unique_words = ' '.join(set(all_words.split(' ')))
        return(unique_words)

    device_categories = device_id_label_category.reset_index().groupby('device_id').agg({'category': get_words}).reset_index()
    device_categories_word = pd.concat([pd.Series(row['device_id'], row['category'].split(' ')) 
                                        for _, row in device_categories.iterrows()]).reset_index()
    device_categories_word.columns = ['word','device_id']
    device_categories_word = device_categories_word[device_categories_word['word'] != '']

    wordencoder = LabelEncoder().fit(device_categories_word['word'])
    device_categories_word['word_id'] = wordencoder.transform(device_categories_word['word'])
    nwords = len(wordencoder.classes_)
    device_categories_word.set_index(['device_id'],inplace=True)

    # inner join between device_categories_word and gender_age_train by device_id, to get number of row of device_id 
    # in the original training set
    merged_train = device_categories_word.merge(gender_age_train[['idtrain']], left_index=True, right_index=True)

    # inner join between device_categories_word and gender_age_test by device_id, to get number of row of device_id 
    # in the original test set
    merged_test = device_categories_word.merge(gender_age_test[['idtest']], left_index=True, right_index=True)

    # get number of training and test examples
    n_train_examples = gender_age_train.shape[0]
    n_test_examples = gender_age_test.shape[0]

    # create sparse matrices with app features
    train_category_word = csr_matrix((np.ones(merged_train.shape[0]),(merged_train['idtrain'],merged_train['word_id'])),
                                      shape=(n_train_examples,nwords))
    test_category_word = csr_matrix((np.ones(merged_test.shape[0]),(merged_test['idtest'],merged_test['word_id'])),
                                      shape=(n_test_examples,nwords))

    return train_category_word, test_category_word