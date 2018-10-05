from base_model import BaseModel
import datetime

from xgboost import XGBClassifier, Booster
from sklearn.preprocessing import LabelEncoder


class DefaultXGBoost(BaseModel):
    def __init__(self, file_name: str = None):
        super().__init__()

        params = {}
        params['learning_rate'] = 0.02
        params['n_estimators'] = 1000
        params['max_depth'] = 4
        params['subsample'] = 0.9
        params['colsample_bytree'] = 0.9

        if file_name:
            self.model_name = file_name
        else:
            datetime_now = datetime.datetime.now().replace(microsecond=0).isoformat().replace(':','-')
            self.model_name = datetime_now + '_default_xgboost.bin'

        self.model = XGBClassifier(**params)

    def save_model(self):
        model_name = self.model_path + self.model_name
        self.model.save_model(model_name)

    def load_model(self):
        model_name = self.model_path + self.model_name
        booster = Booster()
        booster.load_model(model_name)
        self.model._Booster = booster
        self.model._le = LabelEncoder().fit([0., 1.])
