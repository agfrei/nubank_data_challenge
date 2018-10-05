from fraud_xgboost import FraudXGBoost

if __name__ == '__main__':
    print('Predicting: XGBoost for Fraud')
    xgb = FraudXGBoost(saved_model='2018-10-05T17-41-40_fraud_xgboost.bin')
    xgb.predict(file_path='../../data/raw/', file_name='acquisition_test.csv')