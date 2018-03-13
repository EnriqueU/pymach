#!/usr/bin/python
# -*- coding: utf-8 -*-

# Author: Gusseppe Bravo <gbravor@uni.pe>
# License: BSD 3 clause
"""
This module will prepare the dataset, i.e, modify the data. Thoughts:
    - Applying scaling, normalizing, cleaning, etc.
    - Processing will be performed depending on the inferred algorithm.
and so forth.
"""

from __future__ import print_function

__all__ = ['pipeline']

import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MinMaxScaler, Normalizer,\
StandardScaler, RobustScaler, LabelEncoder, FunctionTransformer


class Prepare():
    """ A class for data preparation """

    def __init__(self, definer):
        self.problem_type = definer.problem_type

    def pipeline(self):
        """ This function chooses the best way to scale a data"""
        transformers = []
        scaler = RobustScaler()
        normalizer = Normalizer()
        transformers.append(('scaler', scaler))
        transformers.append(('normalizer', normalizer))
        return FeatureUnion(transformers)

    """
    class CategoricalToNumeric(BaseEstimator, TransformerMixin):
        # A class for parsing categorical columns

        def categoricalColumns(self, df):
            cols = df.columns
            cols_numeric = df._get_numeric_data().columns
            return list(set(cols) - set(cols_numeric))

        def categoricalToNumeric(self, df):
            cat_columns = self.categoricalColumns(df)
            if cat_columns:
                self.categoricalData = True
                for category in cat_columns:
                    encoder = LabelEncoder()
                    #df.loc[:, category+'_n'] = encoder.fit_transform(df[category])
                    df.loc[:, category] = encoder.fit_transform(df[category])

            #df.drop(cat_columns, axis=1, inplace=True)
            return df

        def transform(self, X, y=None, **transform_params):
            #X = pd.DataFrame(X)
            return self.categoricalToNumeric(X)
            #return X.dropna()

        def fit_transform(self, X, y=None, **fit_params):
            self.fit(X, y, **fit_params)
            return self.transform(X)

        def fit(self, X, y=None, **fit_params):
            return self
    """
