# -*- coding: utf-8 -*-
"""Util functions."""
import glob
import os

def get_model_name(suffix: str):
    """Get the last trainned model by suffix.

    Args:
        - suffix: the file suffix to search model.

    Returns:
        The mos recent model name
    
    """
    model_name = sorted(
        glob.glob(os.path.join('models/', suffix)))[-1]
    model_name = model_name.replace('models/', '')

    return model_name

def run_model(model_class, model_suffix:str, file_path: str = 'data/raw/', file_name: str = 'acquisition_test.csv'):
    """Run a specific model.

    Args:
        - model_class:
        - model_suffix:
        - file_path:
        - file_name:

    Returns:
        Model predictions

    """
    model_name = get_model_name(model_suffix)
    model = model_class(saved_model=model_name)
    pred = model.predict(file_path=file_path, file_name=file_name)
    return pred
    