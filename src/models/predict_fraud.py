"""Make predictions for fraud using ensemble."""
import glob
import os
import sys
from model_fraud_ensemble import FraudEnsemble

if __name__ == '__main__':
    model_name = None

    # check if a model was passed via command line
    if len(sys.argv) >= 2:
        model_name = sys.argv[1]

    # get the most recent model
    if model_name == None:
        model_name = sorted(
            glob.glob(os.path.join('models/', '*_fraud_ensemble.bin')))[-1]
        model_name = model_name.replace('models/', '')

    print('Predicting: Ensemble for Fraud - Using model: {}'.format(
        model_name))

    # make a prediction
    model = FraudEnsemble(saved_model=model_name)
    model.predict(file_path='data/raw/', file_name='acquisition_test.csv')
