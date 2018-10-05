from fraud_xgboost import FraudXGBoost

if __name__ == '__main__':
    print('Trainnig: XGBoost for Fraud')
    xgb = FraudXGBoost()
    xgb.train(file_name='acquisition_train.csv')
