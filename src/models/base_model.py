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


# TODO: Olhar os paths de interin, raw e processed com cuidado
class BaseModel(object):
    def __init__(self):
        self.path = '../../data/processed/'
        # TODO: Revisar os dois
        self.train_file = 'acquisition_train.csv'
        self.prep_train_file = 'acquisition_train_prep.csv'

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

    # TODO: returar o prep daqui
    # TODO: criar o pipeline
    # TODO: # import ast #literal_eval() para eval do latlon e do tags
    def prep(self, df: pd.DataFrame, prep_file_path: str = None, prep_file_name: str = None):
        start = perf_counter()

        prep_file_path = prep_file_path or self.path
        prep_file_name = prep_file_name or self.prep_train_file
        prep_file = prep_file_path + prep_file_name

        # TODO: Verificar se vai sempre fazer isso mesmo
        # if os.path.isfile(prep_file):
        #     end = perf_counter()
        #     print('Prep time elapsed: {}'.format(end - start))
        #     return pd.read_csv(prep_file)

        # removing fraud from our analysis
        # TODO: Verificar isso
        if 'target_fraud' in df.columns:
            df = df[df['target_fraud'].isnull()]

        # drop missing values on target_default
        if 'target_default' in df.columns:
            df.dropna(subset=['target_default'], inplace=True)
            # df = df['target_default'].astype('int', copy=False)

        # Drop columns wich will not be used for our model
        # TODO: explicar todas
        drop_cols = [
            'ids',
            'credit_limit',
            'channel',
            'external_data_provider_first_name',
            'profile_phone_number',
            'target_fraud',
            'facebook_profile',
            'profile_tags',
            'user_agent'
        ]
        for col in drop_cols:
            if col in df.columns:
                df.drop(col, axis=1, inplace=True)

        # Bool columns
        df.applymap(lambda x: 1 if x else 0)

        # Encoding categorical columns
        # TODO: melhorar essa abordagem
        # TODO: latlon
        # TODO: user agent é muito importante, **VALIDAR**
        # TODO: profile tags - hashing
        encoding_cols = [
            'score_1', 'score_2', 'reason',
            'state', 'zip', 'job_name', 'real_state',
            'application_time_applied', 'email',
            'lat_lon', 'marketing_channel', 'shipping_state',
            'shipping_zip_code'
        ]
        df = self.encode(df, encoding_cols)

        # for c in df.columns:
        #     if df[c].dtype == 'object':
        #         print('***** {} *****'.format(c))
        #         # df = self.encode(df, [c])

        # Missing values and inf
        # TODO: melhorar essa abordagem
        df.replace([np.inf, -np.inf], np.nan)
        df.fillna(-1, inplace=True)

        # Boolean cols
        # print(df['facebook_profile'].dtype)
        # df = df['facebook_profile'].astype('int', copy=False)
        # # df = df['facebook_profile'].apply(lambda x: 1 if x else 0)
        # print(df['facebook_profile'].dtype)
        # exit()

        # df.to_csv(prep_file)

        end = perf_counter()
        print('Prep time elapsed: {}'.format(end - start))
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

    def train(self, file_name: str, file_path: str = None):
        start = perf_counter()

        file_path = file_path or self.path
        train_file = os.path.join(file_path, file_name)

        # Reading and preparing the dataset
        df = pd.read_csv(train_file)
        df = df[:100]
        df = self.prep(df)

        # Split X and y
        X = df.drop('target_default',axis=1).values.astype(np.float)
        y = df['target_default'].values.astype(np.float)

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

