from model_spend_svr import SpendSVR

# TODO: pegar via linha de comando o modelo a ser usado, ou pegar o último caso não passe nenhum
if __name__ == '__main__':
    print('Predicting: SVR for Spend')
    svr = SpendSVR(saved_model='2018-10-06T03-33-30_spend_svr.bin')
    svr.predict(file_path='data/raw/', file_name='acquisition_test.csv')
