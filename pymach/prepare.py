#!/usr/bin/python
# -*- coding: utf-8 -*-

# Author: Gusseppe Bravo <gbravor@uni.pe>
# License: BSD 3 clause
"""
This module provides a few of useful functions (actually, methods)
for preparing the dataset which is to be studied.

"""
from __future__ import print_function
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.pipeline import FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin

__all__ = [
    'pipeline']


class Prepare():
    """ A class for data preparation """

    data = None

    def __init__(self, typeModel='class', typeAlgorithm=''):
        self.typeModel = typeModel
        self.typeAlgorithm = typeAlgorithm

    def pipeline(self):
        transformers = []

        clean = self.Clean()
        transformers.append(('clean', clean))

        if typeAlgorithm in ["NeuralN", "K-N"]:
            minmax = MinMaxScaler(feature_range=(0,1))
            normalizer = Normalizer()
            transformers.append(('minmax', minmax))
            transformers.append(('normalizer', normalizer))
        elif typeAlgorithm in ["LinearR", "LogisticR"]:
            scaler = StandardScaler()
            transformers.append(('scaler', scaler))
        else:
            scaler = StandardScaler()
            transformers.append(('scaler', scaler))

        #binarizer = Binarizer()
        return FeatureUnion(transformers)

    class Clean(TransformerMixin):
        """ A class for removing NAN values """

        def transform(self, X, **transform_params):
            return pandas.DataFrame(X).dropna()

        def fit(self, X, y=None, **fit_params):
            return self



    #def reescale(self):
        #X = Prepare.data.values[:, 0:len(Prepare.data.columns)-1]
        ##Y = Prepare.data.values[:, len(data.columns)-1]

        #scaler = MinMaxScaler(feature_range=(0,1))
        #rescaledX = scaler.fit_transform(X)

        #return rescaledX, scaler

    #def standardize(self):
        #X = Prepare.data.values[:, 0:len(Prepare.data.columns)-1]
        ##Y = Prepare.data.values[:, len(data.columns)-1]

        #scaler = StandardScaler()
        #rescaledX = scaler.fit_transform(X)

        #return rescaledX, scaler

    #def normalize(self):
        #X = Prepare.data.values[:, 0:len(Prepare.data.columns)-1]
        ##Y = Prepare.data.values[:, len(data.columns)-1]

        #normalizer = Normalizer()
        #normalizedX = normalizer.fit_transform(X)

        #return normalizedX, normalizer

    #def binarize(self):
        #X = Prepare.data.values[:, 0:len(Prepare.data.columns)-1]
        ##Y = Prepare.data.values[:, len(data.columns)-1]

        #binarizer = Binarizer()
        #binaryX = binarizer.fit_transform(X)

        #return binaryX, binarizer

    def labelEncoder(self):
        """If a dataset has categorical variables, change it"""
