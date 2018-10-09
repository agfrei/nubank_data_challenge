# -*- coding: utf-8 -*-
"""Main file. Decides whether or not to approve a customer."""

from utils import run_model
import numpy as np
import pandas as pd
from model_fraud_ensemble import FraudEnsemble
from model_default_ensemble import DefaultEnsemble
from model_spend_dtr import SpendDTR

def pre_approve_limit(row):
    """Define pre-approved and limit.

    Parameters:
        - row: pandas dataframe row

    Return:
        A modified pandas dataframe row with calculated columns

    """
    row['approve'] = '2' if row['fraud'] < 0.015 and row['default'] < 0.3 else '0'    
    limit = row['spend_score'] * (1 - row['default']) if row['approve'] == '2' else 0.
    row['limit_float'] = limit
    return row

def approve(row):
    """Aprove a customer.

    Parameters:
        - row: pandas dataframe row

    Return:
        A modified pandas dataframe row with calculated columns

    """
    row['approve'] = '1'
    row['limit'] = '%d' % int(row['limit_float'])
    return row

def deny(row):
    """Deny a customer.

    Parameters:
        - row: pandas dataframe row

    Return:
        A modified pandas dataframe row with calculated columns

    """
    row['approve'] = '0'
    row['limit'] = '%s' % np.nan
    return row

if __name__ == '__main__':
    # max client aproving
    max_approved = 1500

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
    approving['limit_float'] = 0.0
    approving = approving.apply(pre_approve_limit, axis=1)
    
    # filtering the decision
    approved = approving.sort_values(by='limit_float', ascending=False)[:max_approved].apply(approve, axis=1)
    denied = approving.sort_values(by='limit_float', ascending=False)[max_approved:].apply(deny, axis=1)
    approving = pd.concat([approved, denied], axis=0)
    
    approving.drop(['fraud'], axis=1, inplace=True)
    approving.drop(['default'], axis=1, inplace=True)
    approving.drop(['spend_score'], axis=1, inplace=True)
    approving.drop(['limit_float'], axis=1, inplace=True)

    # writing a file
    approving.to_csv('deliverable/approve_limit/approve_limit_submission.csv')
    