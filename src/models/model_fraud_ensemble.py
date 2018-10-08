# -*- coding: utf-8 -*-
"""Model for predict probability of fraud."""
from base_model import BaseModel
from ensemble import Ensemble
from preprocessing import Prep

import datetime
import os

from xgboost import XGBClassifier, Booster
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

import pandas as pd
import numpy as np
import ast
from time import perf_counter


class FraudEnsemble(BaseModel):
    """Ensemble model for fraud."""

    def __init__(self, saved_model: str = None):
        """Create a new object.

        Args:
            - saved_model (str optional): load a pre-treined model if `saved_name` is not None

        """
        super().__init__()

        # Creating a XGBoost model for stacking
        xgb_params = {}
        xgb_params['learning_rate'] = 0.01
        xgb_params['n_estimators'] = 750
        xgb_params['max_depth'] = 6
        xgb_params['colsample_bytree'] = 0.6
        xgb_params['min_child_weight'] = 0.6
        xgb_model = XGBClassifier(**xgb_params)

        # Creating a random forest model for stacking
        rf_params = {}
        rf_params['n_estimators'] = 200
        rf_params['max_depth'] = 6
        rf_params['min_samples_split'] = 70
        rf_params['min_samples_leaf'] = 30
        rf_model = RandomForestClassifier(**rf_params)

        # Creating a Logist Regression model to act as a stacker of other base models
        log_model = LogisticRegression()

        # Creating the stack
        stack = Ensemble(
            n_splits=3, stacker=log_model, base_models=(rf_model, xgb_model))

        # To use as a prefix of model and processed dataset
        self.datetime_prefix = datetime.datetime.now().replace(
            microsecond=0).isoformat().replace(':', '-')

        # Loads a saved model or create a new one
        if saved_model:
            self.model_name = saved_model
        else:
            self.model_name = self.datetime_prefix + '_fraud_ensemble.bin'

        # The final model
        self.model = stack
        print('Model: {}'.format(self.model_name))

    def prep_lat_long(self, df):
        """Apply a preprocessing into lat_lon column."""
        df['lat'] = df['lat_lon'].apply(lambda x: ast.literal_eval(x)[0])
        df['lon'] = df['lat_lon'].apply(lambda x: ast.literal_eval(x)[1])
        return df

    def prep_fraud(self, df):
        """Transform all types of fraud in only one."""
        # On test df this columns doesn't exist
        if 'target_fraud' in df.columns:
            df['target_fraud'].fillna('this_is_not_fraud', inplace=True)
            df['target_fraud'] = df['target_fraud'].apply(lambda x: 0 if x == 'this_is_not_fraud' else 1)
        return df

    def prep(self,
             df: pd.DataFrame,
             prep_file_name: str,
             prep_file_path: str = 'data/processed/'):
        """Preprocess the features.

        Args:
            - df: pandas DataFrame
            - prep_file_name: name of processed file to store
            - prep_file_path: path to store a processed file. Default: data/processed/

        Returns:
            A pandas dataframe processed

        """
        start = perf_counter()

        drop_cols = [
            'ids', 'credit_limit', 'channel', 'reason', 'job_name', 'reason'
            'external_data_provider_first_name', 'profile_phone_number',
            'avg_spend', 'target_default', 'facebook_profile', 'profile_tags',
            'last_amount_borrowed', 'last_borrowed_in_months',
            'zip', 'email', 'user_agent', 'n_issues',
            'application_time_applied', 'application_time_in_funnel',
            'external_data_provider_credit_checks_last_2_year',
            'external_data_provider_credit_checks_last_month',
            'external_data_provider_credit_checks_last_year',
            'external_data_provider_first_name',
            'class', 'member_since', 'credit_line',
            'total_spent', 'total_revolving', 'total_minutes', 
            'total_card_requests', 'total_months', 'total_revolving_months'
        ]

        encoding_cols = [
            'score_1', 'score_2', 'reason', 'state', 'job_name',
            'real_state', 'marketing_channel', 'shipping_state', 
            'shipping_zip_code'
        ]

        null_mean_cols = [
            'ok_since', 'reported_income',
            'external_data_provider_email_seen_before'
        ]

        null_neg_cols = ['n_bankruptcies', 'n_defaulted_loans']

        prep = Prep(df) \
            .apply_custom(self.prep_fraud) \
            .drop_cols(drop_cols) \
            .fill_null_with('mean', null_mean_cols) \
            .fill_null_with(-1, null_neg_cols) \
            .fill_null_with('NA', ['marketing_channel']) \
            .fill_null_with('(0,0)', ['lat_lon']) \
            .apply_custom(self.prep_lat_long) \
            .drop_cols(['lat_lon']) \
            .drop_nulls() \
            .encode(encoding_cols)
        # TODO: .one_hot_encode(exclude=['target_default'])

        df = prep.df

        prep_file_name = self.datetime_prefix + '_' + prep_file_name
        prep_file = os.path.join(prep_file_path, prep_file_name)
        df.to_csv(prep_file)

        end = perf_counter()
        print('Prep time elapsed: {}'.format(end - start))
        return df

    def train(self, file_name: str, file_path: str = None):
        """Train the model calling the `train` method of super class.
        
        Args:
            - file_name (str): The CSV filename used to train the model
            - file_path (str optional): The path to the `file_name`. Default to `data/interim`
        """
        super().train(
            self.prep,
            file_name=file_name,
            file_path=file_path,
            fit_method='fit',
            target_col='target_fraud')

    def predict(self, file_name: str, file_path: str = None):
        """Make a prediction calling the `predict` method of super class.
        
        Args:
            - file_path (str): The path to the `file_name`
            - file_name (str): The CSV file name with features to predict using a saved model
        
        Returns:
            The prediction
            
        """
        pred = super().predict(
            self.prep,
            file_name=file_name,
            file_path=file_path,
            output_file_name='fraud_submission.csv',
            output_path='deliverable/fraud/',
            output_target_col='fraud',
            output_format='%.04f',
            predict_method='predict')

        return pred
