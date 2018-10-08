# -*- coding: utf-8 -*-
"""Main file. Decides whether or not to approve a customer."""

from utils import run_model
import numpy as np
import pandas as pd
from model_fraud_ensemble import FraudEnsemble
from model_default_ensemble import DefaultEnsemble
from model_spend_dtr import SpendDTR

def approve_limit(row):
    """Define approved and limit.

    Parameters:
        - row: pandas dataframe row

    Return:
        A modified pandas dataframe row with calculated columns

    """
    row['approve'] = '1' if row['fraud'] < 0.1 and row['default'] < 0.1 else '0'
    limit = '%d' % int(row['spend_score'] * (1 - row['default'])) if row['approve'] == '1' else '%s' % np.nan
    row['limit'] = limit
    return row

if __name__ == '__main__':
    # checking for fraud
    fraud_pred = run_model(FraudEnsemble, model_suffix='*_fraud_ensemble.bin')
    fraud_pred.set_index(['ids'], inplace=True)
    fraud_pred['fraud'] = fraud_pred['fraud'].apply(pd.to_numeric)
    
    # checking for default
    default_pred = run_model(DefaultEnsemble, model_suffix='*_default_ensemble.bin')
    default_pred.set_index(['ids'], inplace=True)
    default_pred['default'].astype(float, inplace=True)
    default_pred['default'] = default_pred['default'].apply(pd.to_numeric)

    # checking for spend
    spend_pred = run_model(SpendDTR, model_suffix='*_spend_dtr.bin')
    spend_pred.set_index(['ids'], inplace=True)
    spend_pred['spend_score'].astype(int, inplace=True)
    spend_pred['spend_score'] = spend_pred['spend_score'].apply(pd.to_numeric)

    # making decision
    approving = pd.concat([fraud_pred, default_pred, spend_pred], axis=1)
    approving['approve'] = '0'
    approving['limit'] = np.nan
    approving = approving.apply(approve_limit, axis=1)
    approving.drop(['fraud'], axis=1, inplace=True)
    approving.drop(['default'], axis=1, inplace=True)
    approving.drop(['spend_score'], axis=1, inplace=True)
    
    # writing a file
    approving.to_csv('deliverable/approve_limit/approve_limit_submission.csv')
    