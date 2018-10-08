"""Train a model for predict fraud."""
from model_fraud_ensemble import FraudEnsemble

if __name__ == '__main__':
    print('Trainnig: Ensemble for Fraud')
    model = FraudEnsemble()
    model.train(file_name='acquisition_train.csv')
