import feature_engineering as fe
from load_data import load_data

if __name__ == "__main__":

    app_events, app_labels, events, gender_age_train, gender_age_test, label_categories, brand_model = load_data()

    #################################################################
    # Feature engineering
    #################################################################  

    # Create one-hot encoding of brand and model
    train_brand, test_brand, train_model, test_model = fe.one_hot_brand_model(brand_model, gender_age_train, gender_age_test)

    # Create weekday and hour features (represented using one-hot encoding)
    train_weekday, train_hour, train_weekday_hour, test_weekday, test_hour, test_weekday_hour = fe.weekday_hour_features(events, gender_age_train, gender_age_test)



                                                                                


