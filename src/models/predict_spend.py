"""Make predictions for spend using ensemble."""
import glob
import os
import sys
from model_spend_dtr import SpendDTR

if __name__ == '__main__':
    model_name = None

    # check if a model was passed via command line
    if len(sys.argv) >= 2:
        model_name = sys.argv[1]

    # get the most recent model
    if model_name == None:
        model_name = sorted(
            glob.glob(os.path.join('models/', '*_spend_dtr.bin')))[-1]
        model_name = model_name.replace('models/', '')

    print('Predicting: Decision Tree Regressor for Spend - Using model: {}'.format(
        model_name))

    # make a prediction
    model = SpendDTR(saved_model=model_name)
    model.predict(file_path='data/raw/', file_name='acquisition_test.csv')
