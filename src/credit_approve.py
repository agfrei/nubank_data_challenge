# -*- coding: utf-8 -*-
"""Main file. Decides whether or not to approve a customer."""

import glob
import os
from src.models.model_fraud_ensemble import FraudEnsemble
from src.models.model_default_ensemble import DefaultEnsemble

if __name__ == '__main__':
    # get the most recent fraud model
    if fraud_model_name == None:
        fraud_model_name = sorted(
            glob.glob(os.path.join('models/', '*_fraud_ensemble.bin')))[-1]
        fraud_model_name = fraud_model_name.replace('models/', '')

    print('Predicting: Ensemble for Fraud - Using model: {}'.format(
        model_name))

    # run a fraud predict
    fraud_model = FraudEnsemble(saved_model=model_name)
    fraud_pred = fraud_model.predict(file_path='data/raw/', file_name='acquisition_test.csv')
    print(fraud_pred)
