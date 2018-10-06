from model_default_xgboost import DefaultXGBoost

# TODO: pegar via linha de comando o modelo a ser usado, ou pegar o último caso não passe nenhum
if __name__ == '__main__':
    print('Predicting: XGBoost for Default')
    xgb = DefaultXGBoost(saved_model='2018-10-06T03-23-50_default_xgboost.bin')
    xgb.predict(file_path='data/raw/', file_name='acquisition_test.csv')
