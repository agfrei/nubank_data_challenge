# -*- coding: utf-8 -*-
"""This module implements a BaseModel class."""
import collections
import numpy as np
import pandas as pd
import gc
import ast
import os
import pickle
from time import perf_counter

from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score

import warnings
import logging


class BaseModel(object):
    """Base class do all models.

    All models should inherit from this class
    """

    def __init__(self):
        """Call this on child class."""
        self.path = 'data/interim/'
        self.model_path = 'models/'

        # All this attributes setted to None
        # need to be setted in child class
        self.model = None
        self.model_name = None
        self.logger = None

        # Some pandas useful options
        pd.options.mode.chained_assignment = None
        pd.options.mode.use_inf_as_na = True

        # Ignore some warnings
        warnings.simplefilter('ignore', DeprecationWarning)

    def _init_log(self):
        """Create a log for trainning session.

        All logs are saved on `models/logs` folder.
        Log name is the same same of model object 
        with .log extension.
        """
        self.logger = logging.getLogger(self.model_name.replace('.bin', ''))
        self.logger.setLevel(logging.INFO)

        # create file handler
        fh = logging.FileHandler(
            os.path.join(self.model_path, 'logs/',
                         self.model_name.replace('.bin', '.log')))
        fh.setLevel(logging.INFO)

        # create console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)

        # create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)

        # add the handlers to the logger
        self.logger.addHandler(fh)
        self.logger.addHandler(ch)

    def cross_val_model(self,
                        X,
                        y,
                        n_splits=3,
                        scoring='roc_auc',
                        model_type='clas'):
        """Fit and evaluate a model.

        The train logs are stored in `models/logs` folder.

        Args:
            - X (DataFrame): A pandas dataframe representing the features fo trainning
            - y (DataFrame): Real values to predict
            - n_splits (int): Represents k on k folds
            - scoring (str or tuple): scoring metrics
            - model_type (str): 'clas' for Classification models or 'reg' for Regression models
        
        
        """
        start = perf_counter()

        self._init_log()
        X = np.array(X.astype('float32'))
        y = np.array(y.astype('float32'))

        if model_type == 'reg':
            folds = list(
                KFold(n_splits=n_splits, shuffle=True, random_state=7).split(
                    X, y))
        else:
            folds = list(
                StratifiedKFold(
                    n_splits=n_splits, shuffle=True, random_state=7).split(
                        X, y))

        for j, (train_idx, test_idx) in enumerate(folds):
            start_fold = perf_counter()
            X_train = X[train_idx]
            y_train = y[train_idx]
            X_holdout = X[test_idx]
            y_holdout = y[test_idx]

            self.logger.info("Fit {} fold {}".format(
                str(self.model).split('(')[0], j + 1))
            self.model.fit(X_train, y_train)

            if model_type == 'clas':
                self.logger.info("    y train: ", collections.Counter(y_train))
                self.logger.info("    y test:  ",
                                 collections.Counter(y_holdout))

            if isinstance(scoring, tuple):
                cross_score = cross_validate(
                    self.model, X_holdout, y_holdout, cv=3, scoring=scoring)
                self.logger.info("    Fit Time:   ", cross_score['fit_time'])
                self.logger.info("    Score Time: ", cross_score['score_time'])
                for s in scoring:
                    self.logger.info("    {} test cross_score: {:.5f}".format(
                        s, cross_score['test_' + s].mean()))
                    self.logger.info("    {} train cross_score: {:.5f}".format(
                        s, cross_score['train_' + s].mean()))
            else:
                cross_score = cross_val_score(
                    self.model, X_holdout, y_holdout, cv=3, scoring=scoring)
                self.logger.info("    cross_score: {:.5f}".format(
                    cross_score.mean()))

            if model_type == 'clas':
                y_pred = cross_val_predict(
                    self.model, X_holdout, y_holdout, cv=3)
                conf_mat = confusion_matrix(y_holdout, y_pred)
                self.logger.info(conf_mat)

    def save_model(self):
        """Save a model using pickle.
        
        The model is saved on `models` folder.
        """
        model_name = self.model_path + self.model_name
        with open(model_name, 'wb') as model_file:
            pickle.dump(self.model, model_file)

    def load_model(self):
        """Load a saved model using pickle.
        
        The model is loaded from `models` folder.
        """
        model_name = self.model_path + self.model_name
        with open(model_name, 'rb') as model_file:
            self.model = pickle.load(model_file)

    def train(self,
              prep,
              file_name: str,
              file_path: str = None,
              fit_method: str = None,
              target_col: str = 'target',
              model_type: str = 'clas'):
        """Orchestrate the model trainning.

        Args:
            - prep (callable function): A function to evaluate preprocessing on data
            - file_name (str): The CSV filename used to train the model
            - file_path (str optional): The path to the `file_name`. Default to `data/interim`
            - fit_method (str optional): If None, uses `cros_val_model`, else uses this method of the model directly
            - target_col (str optional): The name of the target col in `file_name`. Default to `target`
            - model_type (str): 'clas' for Classification models or 'reg' for Regression models
        """
        start = perf_counter()

        file_path = file_path or self.path
        train_file = os.path.join(file_path, file_name)

        # Reading and preparing the dataset
        df = pd.read_csv(train_file)  # TODO: REMOVER O SLICING
        df = prep(df, prep_file_name=file_name)

        # Split X and y
        X = df.drop(target_col, axis=1)
        print(df.columns)  # TODO:
        X = X.values.astype(np.float)
        y = df[target_col].values.astype(np.float)

        # train and save the model
        if fit_method == None:
            self.cross_val_model(X, y)
        else:
            eval('self.model.{}(X, y)'.format(fit_method))

        self.save_model()

        end = perf_counter()
        print('Train time elapsed: {}'.format(end - start))

    def predict(self,
                prep,
                file_path: str,
                file_name: str,
                output_path: str,
                output_file_name: str,
                id_col: str = 'ids',
                output_target_col: str = 'target',
                output_format: str = '{}',
                predict_method: str = 'predict_proba'):
        """Generate predictions and save into a formatted file.

        Args:
            - prep (callable function): A function to evaluate preprocessing on data
            - file_path (str): The path to the `file_name`
            - file_name (str): The CSV file name with features to predict using a saved model
            - output_path (str): The path to the `output_file_name`
            - output_file_name (str): The output formatted file name wich will have the generated predictions
            - id_col (str optional): The name of the id col in `file_name`. Default to `ids`
            - target_col (str optional): The name of the target col in `file_name`. Default to `target`
            - output_target_col (str optional): The name of the target col in `output_file_name`. Default to `target`
            - output_format (str optional): The string to pass to .format string method to format the prediction inside output file
            - predict_method (str optional): method name to eval a prediction. Default to `predict_proba`

        """
        start = perf_counter()

        predict_file = os.path.join(file_path, file_name)

        # Reading and preparing the dataset
        df = pd.read_csv(predict_file)  # TODO: REMOVER O SLICING
        ids = df[id_col].values
        df = prep(df, prep_file_name=file_name)
        print(df.columns)  # TODO:
        X = df.values.astype(np.float)

        # load pretrained model into self.model
        self.load_model()

        pred = eval('self.model.{}(df)'.format(predict_method))
        if predict_method == 'predict_proba':  # TODO: predict proba????
            pred = [output_format.format(p[1]) for p in pred]

        df_pred = pd.DataFrame(
            data=list(zip(ids, pred)), columns=[id_col, output_target_col])
        print(df_pred)  # TODO:
        df_pred.to_csv(
            os.path.join(output_path, output_file_name), index=False)

        end = perf_counter()
        print('Predict time elapsed: {}'.format(end - start))
        return df_pred
