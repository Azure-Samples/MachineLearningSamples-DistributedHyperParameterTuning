import pandas as pd
from azure.storage.blob import BlockBlobService
import re

def load_data():

    # Define storage parameters 
    #[comment: is this public or should this be passed in as parameters?]
    ACCOUNT_NAME = "viennatesting"
    ACCOUNT_KEY = "Gc+sxnQOCpdha3ombGfkzsM3+liy4Wh4mgITTv9+CHrQ+CJHosJZ8VeAEqV/Ufx+hUp60BDcaPRf9axfhuZUPQ=="
    CONTAINER_NAME = "dataset"

    # Define blob service     
    my_service = BlockBlobService(account_name=ACCOUNT_NAME, account_key=ACCOUNT_KEY)

    # Load blob
    my_service.get_blob_to_path(CONTAINER_NAME, 'app_events.csv.zip', 'app_events.csv.zip')
    my_service.get_blob_to_path(CONTAINER_NAME, 'app_labels.csv.zip', 'app_labels.csv.zip')
    my_service.get_blob_to_path(CONTAINER_NAME, 'events.csv.zip', 'events.csv.zip')
    my_service.get_blob_to_path(CONTAINER_NAME, 'gender_age_train.csv.zip', 'gender_age_train.csv.zip')
    my_service.get_blob_to_path(CONTAINER_NAME, 'gender_age_test.csv.zip', 'gender_age_test.csv.zip')
    my_service.get_blob_to_path(CONTAINER_NAME, 'label_categories.csv.zip', 'label_categories.csv.zip')
    my_service.get_blob_to_path(CONTAINER_NAME, 'phone_brand_device_model.csv.zip', 'phone_brand_device_model.csv.zip')

    # load data
    app_events = pd.read_csv('app_events.csv.zip', index_col='event_id')
    app_labels = pd.read_csv('app_labels.csv.zip')
    events = pd.read_csv('events.csv.zip', parse_dates=['timestamp'], infer_datetime_format=True, index_col='event_id')
    gender_age_train = pd.read_csv('gender_age_train.csv.zip', index_col='device_id')
    gender_age_test = pd.read_csv('gender_age_test.csv.zip', index_col='device_id')
    label_categories = pd.read_csv('label_categories.csv.zip')
    brand_model = pd.read_csv('phone_brand_device_model.csv.zip')
    brand_model = brand_model.drop_duplicates('device_id',keep='first').set_index('device_id')

    # extract weekday and hour from event timestamps
    events['weekday'] = events['timestamp'].apply(lambda x: x.weekday())
    events['hour'] = events['timestamp'].apply(lambda x: x.hour)
    events['weekday_hour'] = events[['weekday','hour']].apply(lambda x: str(x['weekday'])+"_"+str(x['hour']), axis=1)

    # clean names of categories
    label_categories['category'] = label_categories['category'].apply(lambda x: re.sub('[()_/]',' ',
                                                                                       str(x).lower().replace('game-','game ')))

    return app_events, app_labels, events, gender_age_train, gender_age_test, label_categories, brand_model