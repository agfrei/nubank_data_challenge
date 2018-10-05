from default_xgboost import DefaultXGBoost

if __name__ == '__main__':
    print('Trainnig: XGBoost for Default')
    xgb = DefaultXGBoost()
    xgb.train(file_name='acquisition_train.csv')
