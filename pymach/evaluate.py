#!/usr/bin/python
# -*- coding: utf-8 -*-

# Author: Gusseppe Bravo <gbravor@uni.pe>
# License: BSD 3 clause
"""
This module provides ideas for evaluating some machine learning algorithms.

"""
from __future__ import print_function
import operator
import warnings
import pickle

from joblib import Parallel, delayed
from math import sqrt
import multiprocessing as mp

import numpy as np
import pandas as pd

import plotly.graph_objs as go
import cufflinks as cf # Needed

from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

from collections import OrderedDict
from plotly.offline.offline import _plot_html

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin

from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import FunctionTransformer

#Algorithms Classifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
#Ensembles algorithms
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import BaggingRegressor

# Algorithms Regression
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
#Ensembles algorithms
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import GradientBoostingRegressor

class Evaluate():
    """ A class for resampling and evaluation """
    def __init__(self, definer, preparer, selector):
        self.definer = definer
        self.preparer = preparer
        self.selector = selector
        self.problem_type = definer.problem_type

        self.plot_html = None
        self.report = None
        self.raw_report = None
        self.best_pipelines = None
        self.pipelines = None
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None

        self.test_size = 0.2
        self.num_folds = 10
        self.seed = 7
        self.scoring = 'accuracy'

    def pipeline(self):
        modelos = ['LinearDiscriminantAnalysis', 'SVC', 'GaussianNB', 'MLPClassifier', 'KNeighborsClassifier'
                   'DecisionTreeClassifier', 'LogisticRegression', 'ExtraTreesClassifier', 'AdaBoostClassifier',
                   'RandomForestClassifier', 'GradientBoostingClassifier', 'VotingClassifier', 'KNeighborsRegressor',
                   'RandomForestRegressor', 'AdaBoostRegressor', 'BaggingRegressor', 'ExtraTreesRegressor',
                   'GradientBoostingRegressor', 'DecisionTreeRegressor', 'MLPRegressor', 'SVR']
        self.build_pipelines(modelos)
        self.split_data(self.test_size, self.seed)
        self.evaluate_pipelines()
        self.set_best_pipelines()

        return self

    def set_models(self, modelos=None):
        rs = 1
        models = []
        if (self.problem_type == "Classification"):
            # LDA : Warning(Variables are collinear)
            if 'LinearDiscriminantAnalysis' in modelos:
                models.append( ('LinearDiscriminantAnalysis', LinearDiscriminantAnalysis()) )
            if 'SVC' in modelos:
                models.append( ('SVC', SVC(random_state=rs)) )
            if 'GaussianNB' in modelos:
                models.append( ('GaussianNB', GaussianNB()) )
            if 'MLPClassifier' in modelos:
                models.append( ('MLPClassifier', MLPClassifier(max_iter=1000,random_state=rs)) )
            if 'KNeighborsClassifier' in modelos:
                models.append( ('KNeighborsClassifier', KNeighborsClassifier()) )
            if 'DecisionTreeClassifier' in modelos:
                models.append( ('DecisionTreeClassifier', DecisionTreeClassifier(random_state=rs)) )
            if 'LogisticRegression' in modelos:
                models.append( ('LogisticRegression', LogisticRegression()) )
            # Bagging and Boosting
            if 'ExtraTreesClassifier' in modelos:
                models.append( ('ExtraTreesClassifier', ExtraTreesClassifier(random_state=rs)) )
            if 'AdaBoostClassifier' in modelos:
                models.append( ('AdaBoostClassifier', AdaBoostClassifier(DecisionTreeClassifier(random_state=rs),random_state=rs)) )
            if 'RandomForestClassifier' in modelos:
                models.append( ('RandomForestClassifier', RandomForestClassifier(random_state=rs)) )
            if 'GradientBoostingClassifier' in modelos:
                models.append( ('GradientBoostingClassifier', GradientBoostingClassifier(random_state=rs)) )
            # Voting
            estimators = []
            estimators.append( ("Voting_GradientBoostingClassifier", GradientBoostingClassifier(random_state=rs)) )
            estimators.append( ("Voting_ExtraTreesClassifier", ExtraTreesClassifier(random_state=rs)) )
            voting = VotingClassifier(estimators)
            if 'VotingClassifier'  in modelos:
                models.append( ('VotingClassifier', voting) )

        elif (self.problem_type == "Regression"):
            if 'KNeighborsRegressor' in modelos:
                models.append( ('KNeighborsRegressor', KNeighborsRegressor()) )
            # sklearn.ensemble: Ensemble Methods
            if 'RandomForestRegressor' in modelos:
                models.append( ('RandomForestRegressor',RandomForestRegressor(random_state=rs))  )
            if 'AdaBoostRegressor' in modelos:
                models.append( ('AdaBoostRegressor', AdaBoostRegressor(random_state=rs)))
            if 'BaggingRegressor' in modelos:
                models.append( ('BaggingRegressor', BaggingRegressor(random_state=rs)))
            if 'ExtraTreesRegressor' in modelos:
                models.append( ('ExtraTreesRegressor', ExtraTreesRegressor(random_state=rs)) )
            if 'GradientBoostingRegressor' in modelos:
                models.append( ('GradientBoostingRegressor', GradientBoostingRegressor(random_state=rs)) )
            #sklearn.tree: Decision Trees
            if 'DecisionTreeRegressor' in modelos:
                models.append( ('DecisionTreeRegressor', DecisionTreeRegressor(random_state=rs)) )
            #sklearn.neural_network: Neural network models
            if 'MLPRegressor' in modelos:
                models.append( ('MLPRegressor', MLPRegressor(max_iter=1000, random_state=rs)) )
            # sklearn.svm: Support Vector Machines
            if 'SVR' in modelos:
                models.append( ('SVR', SVR()) )
        return models

    def build_pipelines(self, modelos=None):
        pipelines = []
        models = self.set_models(modelos)
        for m in models:
            pipelines.append((m[0],
                Pipeline([
                    ('preparer', self.preparer),
                    ('selector', self.selector),
                    m,
                ])
            ))
        self.pipelines = pipelines
        return pipelines

    def split_data(self, test_size=0.20, seed=7):
        """ Need to fill """
        X_train, X_test, y_train, y_test =  train_test_split(self.definer.X, self.definer.y, test_size=test_size, random_state=seed)
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

    def evaluate_model(self, m):
        kfold = KFold(n_splits=self.num_folds, random_state=self.seed)
        if (self.problem_type == "Classification"):
            result = cross_val_score(m, self.X_train, self.y_train, cv=kfold, scoring='accuracy')
        elif (self.problem_type == "Regression"):
            result = cross_val_score(m, self.X_train, self.y_train, cv=kfold, scoring='r2')
        return result

    # Evaluating models
    def evaluate_pipelines(self, ax=None):
        self.report = [["Model", "Mean", "STD"]]
        results = []
        names = []
        m = []
        n = []
        for name, model in self.pipelines:
            n.append(name)
            m.append(model)

        num_cores=mp.cpu_count()
        print("****************** num_cores: ",num_cores," *********************")
        pool = mp.Pool(processes=num_cores)
        r = pool.map(self.evaluate_model,m)
        pool.close()
        i=0
        for cv_results in r:
            print("Modeling ...", n[i])
            mean = cv_results.mean()
            std = cv_results.std()
            d = {'name': n[i], 'values': cv_results, 'mean': round(mean, 5), 'std': round(std, 5)}
            results.append(d)
            self.report.append([n[i], round(mean,5), round(std,5)])
            print("Score ...", mean)
            i = i+1
        print("*****************************************************************")

        self.raw_report = sorted(results, key=lambda k: k['mean'], reverse=True)
        headers = self.report.pop(0)
        df_report = pd.DataFrame(self.report, columns=headers)
        print(df_report)
        self.sort_report(df_report)

    def sort_report(self, report):
        report.sort_values(['Mean'], ascending=[False], inplace=True)
        self.report = report.copy()

    def set_best_pipelines(self):
        alg = list(self.report.Model)[0:2]
        best_pipelines = []

        for p in self.pipelines:
            if p[0] in alg:
                best_pipelines.append(p)

        self.best_pipelines = best_pipelines
        #print(self.best_pipelines)

    def plot_to_html(self, fig):
        plotly_html_div, plotdivid, width, height = _plot_html(
                figure_or_data=fig,
                config="",
                validate=True,
                default_width='90%',
                default_height="100%",
                global_requirejs=False)

        return plotly_html_div

    def plot_models(self):
        """" Plot the algorithms by using box plots"""
        results = self.raw_report
        print(type(self.raw_report))
        data = []
        N = len(results)
        c = ['hsl('+str(h)+',50%'+',50%)' for h in np.linspace(0, 270, N)]

        for i, d in enumerate(results):
            trace = go.Box(
                y=d['values'],
                name=d['name'],
                marker=dict(
                    color=c[i],
                ),
                boxmean='sd'
            )
            data.append(trace)

        text_scatter = go.Scatter(
                x=[d['name'] for d in results],
                y=[d['mean'] for d in results],
                name='score',
                mode='markers',
                text=['Explanation' for _ in results]
        )
        data.append(text_scatter)
        layout = go.Layout(
            #showlegend=False,
            title='Hover over the bars to see the details',
            annotations=[
                dict(
                    x=results[0]['name'],
                    y=results[0]['mean'],
                    xref='x',
                    yref='y',
                    text='Best model',
                    showarrow=True,
                    arrowhead=7,
                    ax=0,
                    ay=-40
                ),
                dict(
                    x=results[-1]['name'],
                    y=results[-1]['mean'],
                    xref='x',
                    yref='y',
                    text='Worst model',
                    showarrow=True,
                    arrowhead=7,
                    ax=0,
                    ay=-40
                )
            ]
        )

        fig = go.Figure(data=data, layout=layout)

        self.plot_html = self.plot_to_html(fig)
        return self.plot_html

    def save_plot(self, path):
        with open(path, "w") as plot:
            plot.write(self.plot_html)

    def save_report(self, path):
        # with open(path, "w") as plot:
        self.report.to_csv(path, index=False)
        # plot.write(valuate.report.to_csv())
