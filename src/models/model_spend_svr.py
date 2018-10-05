from base_model import BaseModel
import datetime

from sklearn.svm import SVR
from sklearn.preprocessing import LabelEncoder

import pandas as pd
import numpy as np
from time import perf_counter


class SpendSVR(BaseModel):
    def __init__(self, saved_model: str = None):
        super().__init__()

        # svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)
        # svr_lin = SVR(kernel='linear', C=1e3)
        # svr_poly = SVR(kernel='poly', C=1e3, degree=2)

        params = {}
        params['kernel'] = 'rbf'
        params['C'] = 1e3
        params['gamma'] = 0.1

        self.prep_train_file = 'acquisition_train_prep.csv'
        self.model_scoring = 'mean_squared_error'

        if saved_model:
            self.model_name = saved_model
        else:
            datetime_now = datetime.datetime.now().replace(microsecond=0).isoformat().replace(':','-')
            self.model_name = datetime_now + '_spend_svr.bin'

        self.model = SVR(**params)

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

        # Drop columns wich will not be used for our model
        # TODO: explicar todas
        drop_cols = [
            'ids',
            'credit_limit',
            'channel',
            'external_data_provider_first_name',
            'profile_phone_number',
            'target_default',
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

    # TODO: Usar o pipeline para resolver isso  
    def train(self, file_name: str, file_path: str = None):
        super().train(self.prep, file_name=file_name, file_path=file_path, target_col='credit_line')
    