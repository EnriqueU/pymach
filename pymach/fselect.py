#!/usr/bin/python
# -*- coding: utf-8 -*-

# Author: Gusseppe Bravo <gbravor@uni.pe>
# License: BSD 3 clause
"""
This module provides a few of useful functions (actually, methods)
for feature selection the dataset which is to be studied.

"""
from __future__ import print_function
import numpy as np
import pandas as pd
import math

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import f_regression
from sklearn.feature_selection import f_classif

from sklearn.ensemble import ExtraTreesClassifier
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectFromModel
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin

# __all__ = [
#     'pipeline']


class Select():
    """ A class for feature selection """

    def __init__(self, definer):
        self.problem_type = definer.problem_type
        self.n_features = definer.n_features

    def pipeline(self):
        """ This function chooses the best way to find features"""
        transformers = []
        if( self.n_features >3 ):
            n_features = math.ceil(self.n_features/2)
            pca = PCA(n_components=n_features, svd_solver='randomized', whiten=True)
            transformers.append(('pca', pca))
            """if (self.problem_type == "Classification"):
                kbest = SelectKBest(score_func=f_classif, k=n_features)
                transformers.append(('kbest', kbest))
            elif(self.problem_type == "Regression"):
                kbest = SelectKBest(score_func=f_regression, k=n_features)
                transformers.append(('kbest', kbest))"""
        else:
            pca = PCA(n_components=self.n_features, svd_solver='randomized', whiten=True)
            transformers.append(('pca', pca))
        return FeatureUnion(transformers)

    class CustomFeature(TransformerMixin):
        """ A custome class for featuring """

        def transform(self, X, **transform_params):
            #X = pd.DataFrame(X)
            return X

        def fit(self, X, y=None, **fit_params):
            return self
