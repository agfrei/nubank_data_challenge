import numpy as np
import pandas as pd
import gc
import ast
import os
import pickle
from time import perf_counter

from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score

from sklearn.preprocessing import LabelEncoder
import warnings


# TODO: Gerar os arquivos
# TODO: Matriz de confusão
# TODO: Olhar os paths de interin, raw e processed com cuidado
class BaseModel(object):
    def __init__(self):
        # TODO: pegar do interin
        self.path = '../../data/processed/'
        # TODO: Revisar os dois
        
        self.model = None
        self.model_name = None
        self.model_path = '../../models/'

        pd.options.mode.chained_assignment = None
        warnings.simplefilter('ignore', DeprecationWarning)

    def encode(self, df: pd.DataFrame, columns: list) -> pd.DataFrame:
        """Summary

        Args:
            columns (list): Description

        Returns:
            pd.DataFrame: Description
        """
        l_e = LabelEncoder()
        for col in columns:
            df[col].fillna('-1', inplace=True)
            df[col] = l_e.fit_transform(df[col])
        return df

    def cross_val_model(self, X, y, n_splits=3):
        start = perf_counter()

        X = np.array(X.astype('float32'))
        y = np.array(y.astype('float32'))

        folds = list(StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=2017).split(X, y))

        for j, (train_idx, test_idx) in enumerate(folds):
            start_fold = perf_counter()
            X_train = X[train_idx]
            y_train = y[train_idx]
            X_holdout = X[test_idx]
            y_holdout = y[test_idx]

            print ("Fit {} fold {}".format(str(self.model).split('(')[0], j+1))
            self.model.fit(X_train, y_train)
            cross_score = cross_val_score(self.model, X_holdout, y_holdout, cv=3, scoring='roc_auc')
            end_fold = perf_counter()
            print("    cross_score: {:.5f} - time elapsed: {}".format(cross_score.mean(), end_fold - start_fold))

        end = perf_counter()
        print('Cross val time elapsed: {}'.format(end - start))

    def save_model(self):
        model_name = self.model_path + self.model_name
        with open(model_name, 'wb') as model_file:
            pickle.dump(self.model, model_file)

    def load_model(self):
        model_name = self.model_path + self.model_name
        with open(model_name, 'rb') as model_file:
            self.model = pickle.load(model_file)

    def train(self, prep, file_name: str, file_path: str = None, target_col: str = 'target'):
        start = perf_counter()

        file_path = file_path or self.path
        train_file = os.path.join(file_path, file_name)

        # Reading and preparing the dataset
        df = pd.read_csv(train_file)
        df = df[:1000]
        df = prep(df)

        # Split X and y
        X = df.drop(target_col,axis=1).values.astype(np.float)
        y = df[target_col].values.astype(np.float)

        # train and save the model
        self.cross_val_model(X, y)
        # pred = self.model.predict(X)
        # print(pred)

        self.save_model()

        end = perf_counter()
        print('Train time elapsed: {}'.format(end - start))

    def predict(self, file_path: str, file_name: str):
        start = perf_counter()

        predict_file = os.path.join(file_path, file_name)

        # Reading and preparing the dataset
        df = pd.read_csv(predict_file)
        df = df[:10]
        df = self.prep(df)

        # load pretrained model into self.model
        self.load_model()
        pred = self.model.predict(df)
        print(pred)

        end = perf_counter()
        print('Predict time elapsed: {}'.format(end - start))

