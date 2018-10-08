# -*- coding: utf-8 -*-
"""Module for common preprocessing tasks."""
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


class Prep(object):
    """Perform preprocessing tasks."""

    def __init__(self, df: pd.DataFrame):
        """Create new object.
        
        Args:
            - df (DataFrame): a pandas dataframe to performs preprocessing tasks.
            Al tasks are performed on a copy of this DataFrame

        """
        self.df = df.copy()

    def apply_custom(self, fn):
        """Apply a custom function to the dataframe.
        
        Args:
            - fn: custom function to apply. Should receive the dataframe and returns the modified dataframe

        Returns:
            self

        """
        self.df = fn(self.df)
        return self

    def drop_nulls(self, cols: list = None):
        """Drop all rows with nulls.

        Args:
            - cols (list): list of columns or None to all dataframe 

        Returns:
            self

        """
        if cols == None:
            self.df.dropna(inplace=True)
        else:
            cols = [c for c in cols if c in self.df.columns]
            self.df.dropna(subset=cols, inplace=True)
        return self

    def drop_not_nulls(self, cols: list):
        """Drop all rows with not null values for each column in cols.

        Args:
            - cols (list): list of columns
        
        Returns:
            self

        """
        cols = [c for c in cols if c in self.df.columns]
        for col in cols:
            self.df = self.df[self.df[col].isnull()]
        return self

    def drop_cols(self, cols: list):
        """Drop all listed columns.

        Args:
            - cols (list): list of cols to drop

        Returns:
            self

        """
        cols = [c for c in cols if c in self.df.columns]
        for col in cols:
            self.df.drop(col, axis=1, inplace=True)
        return self

    def bool_to_int(self, cols: list):
        """Transform bool into 1 and 0.
        
        Args:
            - cols (list): list of cols to transform

        Returns:
            Self

        """
        if cols == None:
            self.df.applymap(lambda x: 1 if x else 0)
        else:
            cols = [c for c in cols if c in self.df.columns]
            for col in cols:
                self.df[col] = self.df[col].apply(lambda x: 1 if x else 0)
        return self

    # TODO: Salvar label encoder em pickle
    def encode(self, cols: list):
        """Encode categorical vars into numeric ones.

        Args:
            - cols (list): list of columns to encode

        Returns:
            Self

        """
        l_e = LabelEncoder()
        cols = [c for c in cols if c in self.df.columns]
        for col in cols:
            self.df[col].fillna('N/A-ENC', inplace=True)
            self.df[col] = l_e.fit_transform(self.df[col])
        return self

    def fill_null_with(self, val, cols=None):
        """Fill all null with a same value.

        Args:
            - val: can be `mean` to replace null with the mean of the columns
            or any value to put in place of nulls.
            - cols (list): list of columns or None to all dataframe 

        Returns:
            self

        """
        if cols == None:
            self.df.fillna(val, inplace=True)
        else:
            cols = [c for c in cols if c in self.df.columns]
            if isinstance(val, str):
                if val == 'mean':
                    for col in cols:
                        self.df[col].fillna((self.df[col].mean()),
                                            inplace=True)
                else:
                    for col in cols:
                        self.df[col].fillna(val, inplace=True)
            else:
                for col in cols:
                    self.df[col].fillna(val, inplace=True)

        return self

    def _OHE_by_unique(self, train, one_hot, limit):
        #ONE-HOT enconde features with more than 2 and less than 'limit' unique values
        df = train.copy()
        for c in one_hot:
            if len(one_hot[c]) > 2 and len(one_hot[c]) < limit:
                for val in one_hot[c]:
                    df[c + '_oh_' + str(val)] = (df[c].values == val).astype(
                        np.int)
        return df

    def one_hot_encode(self, exclude: list = None):
        """One hot encode columns.

        Args:
            exclude: list of columns to not apply one hot encoding

        Returns:
            self

        """
        if exclude == None:
            exclude = []
        one_hot = {
            c: list(self.df[c].unique())
            for c in self.df.columns if c not in exclude
        }
        self.df = self._OHE_by_unique(self.df, one_hot, 7)
        return self
