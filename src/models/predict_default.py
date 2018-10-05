from default_xgboost import DefaultXGBoost

if __name__ == '__main__':
    print('Predicting: XGBoost for Default')
    xgb = DefaultXGBoost(file_name='2018-10-05T16-05-47_default_xgboost.bin')
    xgb.predict(file_path='../../data/raw/', file_name='acquisition_test.csv')