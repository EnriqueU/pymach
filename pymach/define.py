#!/usr/bin/python
# -*- coding: utf-8 -*-

# Author: Gusseppe Bravo <gbravor@uni.pe>
# License: BSD 3 clause

__all__ = [
    'pipeline']

import pandas as pd
import numpy as np

from collections import OrderedDict
from tools import sizeof_file
from sklearn import preprocessing

class Define():
    """Define module.

    Parameters
    ------------
    data_name   : string (The dataset's name which is expected to be a csv file)
    header      : list (The dataset's header, i.e, the features and the class name)
    response    : string (The name of the variable will be used for prediction.)
    problem_type: string (Classification and Regression.)

    Attributes
    -----------
    n_features  : int (number of features or predictors)
    samples     : int (Number of rows in the dataset)

    """
    def __init__(self,
            data_path,data_name,
            problem_type='Classification'):

        self.problem_type = problem_type
        self.data_path = data_path
        self.data_name = data_name
        self.response = 'class'

        self.n_features = None
        self.describe = None
        self.samples = None
        self.size = None
        self.data = None
        self.X_1 = None
        self.X_2 = None
        self.X = None
        self.y = None

    def pipeline(self):
        self.read()
        self.description()
        self.categoricalToNumeric()

        return self

    def read(self):
        self.head_y = None
        self.count = None
        try:
            if self.data_path is not None:
                self.data = pd.read_csv(self.data_path)
                self.count = len(self.data.columns.values) - 1
                self.head_y = self.data.columns.values[self.count]
                self.data.rename(columns={self.head_y:'class'}, inplace=True)
                self.data.dropna(inplace=True)
                self.X = self.data.loc[:, self.data.columns != self.response]
                self.X_1 = self.X
                self.y = self.data.loc[:, self.data.columns == self.response]
                self.y = np.ravel(self.y)
        except:
            print("Error reading")

    def description(self):
        self.n_features = len(self.data.columns)-1
        self.samples = len(self.data)
        self.size = sizeof_file(self.data_path)

        self.describe = [self.data_name.replace(".csv",""), self.n_features, self.samples, self.size]
        self.describe = pd.DataFrame([self.describe], columns = ["name","n_features","samples","size"])
        return self.describe

    def categoricalToNumeric(self):
        if self.X.select_dtypes(include=[object]).shape[1]:
            numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
            self.X_1 = self.X.select_dtypes(include=numerics)
            self.X_2 = self.X.select_dtypes(include=[object])
            le = preprocessing.LabelEncoder()
            self.X_2 = self.X_2.apply(le.fit_transform)
            self.X = pd.concat([self.X_1,self.X_2],axis=1)
