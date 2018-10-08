# -*- coding: utf-8 -*-
"""Train a model for predict spend."""
from model_spend_dtr import SpendDTR

if __name__ == '__main__':
    print('Trainnig: Decision Tree for Spend')
    model = SpendDTR()
    model.train(file_name='acquisition_train.csv')
