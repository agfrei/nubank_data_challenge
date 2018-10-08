# -*- coding: utf-8 -*-
"""Main file. Decides whether or not to approve a customer."""

import glob
import os
import pandas as pd
from model_fraud_ensemble import FraudEnsemble
from model_default_ensemble import DefaultEnsemble

if __name__ == '__main__':
    # CHECKING FOR FRAUD
    # get the most recent fraud model
    fraud_model_name = sorted(
        glob.glob(os.path.join('models/', '*_fraud_ensemble.bin')))[-1]
    fraud_model_name = fraud_model_name.replace('models/', '')

    print('Predicting: Ensemble for Fraud - Using model: {}'.format(
        fraud_model_name))

    # run a fraud predict
    fraud_model = FraudEnsemble(saved_model=fraud_model_name)
    fraud_pred = fraud_model.predict(file_path='data/raw/', file_name='acquisition_test.csv')
    fraud_pred.set_index(['ids'], inplace=True)
    print(fraud_pred)
    
    # CHECKING FOR DEFAULT
    # get the most recent default model
    default_model_name = sorted(
        glob.glob(os.path.join('models/', '*_default_ensemble.bin')))[-1]
    default_model_name = default_model_name.replace('models/', '')

    print('Predicting: Ensemble for Default - Using model: {}'.format(
        default_model_name))

    # run a default predict
    default_model = DefaultEnsemble(saved_model=default_model_name)
    default_pred = default_model.predict(file_path='data/raw/', file_name='acquisition_test.csv')
    default_pred.set_index(['ids'], inplace=True)
    print(default_pred)

    print(pd.concat([fraud_pred, default_pred], axis=1))
    