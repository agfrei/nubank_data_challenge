from model_spend_svr import SpendSVR

if __name__ == '__main__':
    print('Trainnig: SVR for Spend')
    svr = SpendSVR()
    svr.train(file_name='acquisition_train.csv')
