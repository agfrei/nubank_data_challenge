from model_fraud_xgboost import FraudXGBoost

# TODO: pegar via linha de comando o modelo a ser usado, ou pegar o último caso não passe nenhum
if __name__ == '__main__':
    print('Predicting: XGBoost for Fraud')
    xgb = FraudXGBoost(saved_model='2018-10-06T03-29-04_fraud_xgboost.bin')
    xgb.predict(file_path='data/raw/', file_name='acquisition_test.csv')
