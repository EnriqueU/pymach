#!/usr/bin/python
# -*- coding: utf-8 -*-

# Author: Gusseppe Bravo <gbravor@uni.pe>
# License: BSD 3 clause
"""
This module provides ideas for improving some machine learning algorithms.

"""
from __future__ import print_function
import tools

import warnings
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objs as go
warnings.filterwarnings("ignore", category=DeprecationWarning)

from collections import OrderedDict
from time import time
from plotly.offline.offline import _plot_html
from scipy.stats import randint
from scipy.stats import uniform
from scipy.stats import expon

# from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

import multiprocessing as mp



class Improve():
    """ A class for improving """
    bestConfiguration = None
    def __init__(self, evaluator, optimizer, modelos):
        self.evaluator = evaluator
        self.pipelines = evaluator.build_pipelines(modelos)
        self.modelos = modelos
        self.optimizer = optimizer
        self.problem_type = evaluator.problem_type
        self.search = None
        self.score_report = None
        self.full_report = None
        self.best_search = None
        self.best_model = None
        self.cv = 10

    def pipeline(self):
        self.improve_grid_search()
        return self

    def adaboost_paramC(self, method='GridSearchCV'):
        if method == 'GridSearchCV':
            parameters = {
                'selector__pca__svd_solver': ['full', 'arpack', 'randomized'],
                'selector__pca__whiten': [True,False],
                'AdaBoostClassifier__n_estimators': [50, 100],
                'AdaBoostClassifier__learning_rate': [1.0, 2.0]
            }
        elif  method == 'RandomizedSearchCV':
            parameters = {
                'selector__pca__svd_solver': ['full', 'arpack', 'randomized'],
                'selector__pca__whiten': [True,False],
                'AdaBoostClassifier__n_estimators': randint(50,100),
                'AdaBoostClassifier__learning_rate': expon(0,5)
            }
        else:
            pass
        return parameters

    def voting_paramC(self, method='GridSearchCV'):
        if method == 'GridSearchCV':
            parameters = {
                'selector__pca__svd_solver': ['full', 'arpack', 'randomized'],
                'selector__pca__whiten': [True,False],
                'VotingClassifier__voting': ['hard', 'soft']
            }
        elif  method == 'RandomizedSearchCV':
            parameters = {
                'selector__pca__svd_solver': ['full', 'arpack', 'randomized'],
                'selector__pca__whiten': [True,False],
                'VotingClassifier__voting': ['hard', 'soft']
            }
        else:
            pass
        return parameters

    def gradientboosting_paramC(self, method='GridSearchCV'):
        if method == 'GridSearchCV':
            parameters = {
                'selector__pca__svd_solver': ['full', 'arpack', 'randomized'],
                'selector__pca__whiten': [True,False],
                'GradientBoostingClassifier__n_estimators': [200, 250],
                'GradientBoostingClassifier__max_depth': [3,6,9],
                'GradientBoostingClassifier__learning_rate': [0.1, 0.2, 0.3]
            }
        elif  method == 'RandomizedSearchCV':
            parameters = {
                'selector__pca__svd_solver': ['full', 'arpack', 'randomized'],
                'selector__pca__whiten': [True,False],
                'GradientBoostingClassifier__n_estimators': randint(200,250),
                'GradientBoostingClassifier__max_depth': randint(3,9),
                'GradientBoostingClassifier__learning_rate': expon(0,1)
            }
        else:
            pass
        return parameters

    def extratrees_paramC(self, method='GridSearchCV'):
        if method == 'GridSearchCV':
            parameters = {
                'selector__pca__svd_solver': ['full', 'arpack', 'randomized'],
                'selector__pca__whiten': [True,False],
                'ExtraTreesClassifier__n_estimators': [10, 15, 20],
                'ExtraTreesClassifier__criterion': ['gini', 'entropy']
                # 'ExtraTreesClassifier__min_samples_leaf': [1,2,3,4,5],
                # 'ExtraTreesClassifier__min_samples_leaf': range(200,1001,200),
                # 'ExtraTreesClassifier__max_leaf_nodes': [2,3,4,5],
                # 'ExtraTreesClassifier__max_depth': [2,3,4,5]
            }
        elif  method == 'RandomizedSearchCV':
            parameters = {
                'selector__pca__svd_solver': ['full', 'arpack', 'randomized'],
                'selector__pca__whiten': [True,False],
                'ExtraTreesClassifier__n_estimators': [10, 15, 20],
                'ExtraTreesClassifier__criterion': ['gini', 'entropy'],
                'ExtraTreesClassifier__min_samples_leaf': randint(1,6),
                # 'ExtraTreesClassifier__min_samples_leaf': range(200,1001,200),
                 'ExtraTreesClassifier__max_leaf_nodes': randint(2,10),
                 'ExtraTreesClassifier__max_depth': randint(1,10)
            }
        else:
            pass
        return parameters

    def randomforest_paramC(self, method='GridSearchCV'):
        if method == 'GridSearchCV':
            parameters = {
                'selector__pca__svd_solver': ['full', 'arpack', 'randomized'],
                'selector__pca__whiten': [True,False],
                'RandomForestClassifier__n_estimators': [10, 15],
                'RandomForestClassifier__criterion': ['gini', 'entropy'],
                'RandomForestClassifier__warm_start': [True,False]
                # 'RandomForestClassifier__min_samples_leaf': [1,2,3,4,5],
                # 'RandomForestClassifier__max_leaf_nodes': [2,3,4,5],
                # 'RandomForestClassifier__max_depth': [2,3,4,5],
            }
        elif  method == 'RandomizedSearchCV':
            parameters = {
                'selector__pca__svd_solver': ['full', 'arpack', 'randomized'],
                'selector__pca__whiten': [True,False],
                'RandomForestClassifier__min_samples_leaf': randint(1,20),
                'RandomForestClassifier__max_leaf_nodes': randint(2,20),
                'RandomForestClassifier__max_depth': randint(1,20)
                # 'RandomForestClassifier__min_samples_leaf': [1,2,3,4,5],
                # 'RandomForestClassifier__max_leaf_nodes': [2,3,4,5],
                # 'RandomForestClassifier__max_depth': [2,3,4,5],
            }
        else:
            pass
        return parameters

    def decisiontree_paramC(self, method='GridSearchCV'):
        if method == 'GridSearchCV':
            parameters = {
                'selector__pca__svd_solver': ['full', 'arpack', 'randomized'],
                'selector__pca__whiten': [True,False],
                'DecisionTreeClassifier__criterion': ['gini','entropy'],
                'DecisionTreeClassifier__splitter': ['best','random'],
                'DecisionTreeClassifier__max_features': ['sqrt','log2', None]
                # 'DecisionTreeClassifier__max_leaf_nodes': [2,3, None],
                # 'DecisionTreeClassifier__max_depth': [2,3, None],
                # 'DecisionTreeClassifier__min_samples_leaf': [1,3,5, None]

            }
        elif  method == 'RandomizedSearchCV':
            parameters['DecisionTreeClassifier__min_samples_leaf'] = randint(1,20)
            parameters['DecisionTreeClassifier__max_leaf_nodes'] = randint(2,20)
            parameters['DecisionTreeClassifier__max_depth'] = randint(1,20)
        else:
            pass

        return parameters

    def lda_paramC(self, method='GridSearchCV'):
        if method == 'GridSearchCV':
            parameters = {
                'selector__pca__svd_solver': ['full', 'arpack', 'randomized'],
                'selector__pca__whiten': [True,False],
                'LinearDiscriminantAnalysis__solver': ['svd']

            }
        elif  method == 'RandomizedSearchCV':
            pass
        else:
            pass
        return parameters

    def svc_paramC(self, method='GridSearchCV'):
        if method == 'GridSearchCV':
            parameters = {
                'selector__pca__svd_solver': ['full', 'arpack', 'randomized'],
                'selector__pca__whiten': [True,False],
                'SVC__kernel': ['linear','poly', 'rbf','sigmoid'],
                # 'SVC__kernel': ['rbf'],
                'SVC__C': [1, 10, 100],
                'SVC__decision_function_shape': ['ovo','ovr']
                # 'SVC__decision_function_shape': ['ovr']

            }
        elif  method == 'RandomizedSearchCV':
            pass
        else:
            pass
        return parameters

    def knn_paramC(self, method='GridSearchCV'):
        if method == 'GridSearchCV':
            parameters = {
                'selector__pca__svd_solver': ['full', 'arpack', 'randomized'],
                'selector__pca__whiten': [True,False],
                'KNeighborsClassifier__n_neighbors': [5,7,11],
                'KNeighborsClassifier__weights': ['uniform','distance'],
                'KNeighborsClassifier__algorithm': ['ball_tree','kd_tree','brute']
                # 'KNeighborsClassifier__algorithm': ['auto']

            }
        elif  method == 'RandomizedSearchCV':
            parameters['KNeighborsClassifier__n_neighbors'] = randint(5,10)
        else:
            pass
        return parameters

    def logistic_paramC(self, method='GridSearchCV'):
        if method == 'GridSearchCV':
            parameters = {
                'selector__pca__svd_solver': ['full', 'arpack', 'randomized'],
                'selector__pca__whiten': [True,False],
                'LogisticRegression__penalty': ['l2'],
                # 'LogisticRegression__solver': ['newton-cg','lbfgs','liblinear','sag'],
                'LogisticRegression__solver': ['newton-cg','lbfgs', 'sag'],
                'LogisticRegression__warm_start': [True,False]
            }
        elif  method == 'RandomizedSearchCV':
            pass
        else:
            pass
        return parameters

    def naivebayes_paramC(self, method='GridSearchCV'):
        if method == 'GridSearchCV':
            parameters = {
                'selector__pca__svd_solver': ['full', 'arpack', 'randomized'],
                'selector__pca__whiten': [True,False]
                # 'GaussianNB__priors': [None]
            }
        elif  method == 'RandomizedSearchCV':
            pass
        else:
            pass
        return parameters

    def mlperceptron_paramC(self, method='GridSearchCV'):
        if method == 'GridSearchCV':
            parameters = {
                'selector__pca__svd_solver': ['full', 'arpack', 'randomized'],
                'selector__pca__whiten': [True,False],
                'MLPClassifier__hidden_layer_sizes': [100],
                'MLPClassifier__activation': ['identity', 'logistic', 'tanh', 'relu']
            }
        elif  method == 'RandomizedSearchCV':
            pass
        else:
            pass
        return parameters

    ############################# Regression ###################################

    def knn_paramR(self,method='GridSearchCV'):
        if method == 'GridSearchCV':
            parameters = {
                'selector__pca__svd_solver': ['full', 'arpack', 'randomized'],
                'selector__pca__whiten': [True,False],
                'KNeighborsRegressor__n_neighbors': [5,7,11],
                'KNeighborsRegressor__weights': ['uniform','distance'],
                'KNeighborsRegressor__algorithm': ['ball_tree','kd_tree','brute']
                # 'KNeighborsClassifier__algorithm': ['auto']
            }
        elif  method == 'RandomizedSearchCV':
            pass
        else:
            pass
        return parameters

    def randomforest_paramR(self,method='GridSearchCV'):
        if method == 'GridSearchCV':
            parameters = {
                'selector__pca__svd_solver': ['full', 'arpack', 'randomized'],
                'selector__pca__whiten': [True,False],
                'RandomForestRegressor__n_estimators': [10, 15],
                'RandomForestRegressor__criterion': ['mse', 'mae'],
                'RandomForestRegressor__warm_start': [True,False]
                # 'RandomForestClassifier__min_samples_leaf': [1,2,3,4,5],
                # 'RandomForestClassifier__max_leaf_nodes': [2,3,4,5],
                # 'RandomForestClassifier__max_depth': [2,3,4,5],
            }
        elif  method == 'RandomizedSearchCV':
            pass
        else:
            pass
        return parameters

    def adaboost_paramR(self,method='GridSearchCV'):
        if method == 'GridSearchCV':
            parameters = {
                'selector__pca__svd_solver': ['full', 'arpack', 'randomized'],
                'selector__pca__whiten': [True,False],
                'AdaBoostRegressor__n_estimators': [50, 100],
                'AdaBoostRegressor__learning_rate': [0.01,0.05,0.1,0.3,1],
                'AdaBoostRegressor__loss' : ['linear', 'square', 'exponential']
            }
        elif  method == 'RandomizedSearchCV':
            pass
        else:
            pass
        return parameters

    def bagging_paramR(self,method='GridSearchCV'):
        if method == 'GridSearchCV':
            parameters = {
                'selector__pca__svd_solver': ['full', 'arpack', 'randomized'],
                'selector__pca__whiten': [True,False],
                'BaggingRegressor__n_estimators': [50, 100],
                'BaggingRegressor__warm_start': [True,False]
            }
        elif  method == 'RandomizedSearchCV':
            pass
        else:
            pass
        return parameters

    def extratrees_paramR(self,method='GridSearchCV'):
        if method == 'GridSearchCV':
            parameters = {
                'selector__pca__svd_solver': ['full', 'arpack', 'randomized'],
                'selector__pca__whiten': [True,False],
                'ExtraTreesRegressor__n_estimators': [10, 15, 20],
                'ExtraTreesRegressor__criterion': ['mse', 'mae'],
                'ExtraTreesRegressor__warm_start': [True, False]
                # 'ExtraTreesClassifier__min_samples_leaf': [1,2,3,4,5],
                # 'ExtraTreesClassifier__min_samples_leaf': range(200,1001,200),
                # 'ExtraTreesClassifier__max_leaf_nodes': [2,3,4,5],
                # 'ExtraTreesClassifier__max_depth': [2,3,4,5]
            }
        elif  method == 'RandomizedSearchCV':
            pass
        else:
            pass
        return parameters

    def gradientboosting_paramR(self,method='GridSearchCV'):
        if method == 'GridSearchCV':
            parameters = {
                'selector__pca__svd_solver': ['full', 'arpack', 'randomized'],
                'selector__pca__whiten': [True,False],
                'GradientBoostingRegressor__loss': ['ls','lad','huber','quantile'],
                'GradientBoostingRegressor__n_estimators': [100, 200, 250],
                'GradientBoostingRegressor__max_depth': [3,6,9],
                'GradientBoostingRegressor__learning_rate': [0.1, 0.2, 0.3]
            }
        elif  method == 'RandomizedSearchCV':
            pass
        else:
            pass
        return parameters

    def decisiontree_paramR(self,method='GridSearchCV'):
        if method == 'GridSearchCV':
            parameters = {
                'selector__pca__svd_solver': ['full', 'arpack', 'randomized'],
                'selector__pca__whiten': [True,False],
                'DecisionTreeRegressor__criterion': ['mse','friedman_mse','mae'],
                'DecisionTreeRegressor__splitter': ['best','random'],
                'DecisionTreeRegressor__max_features': ['sqrt','log2', None]
                # 'DecisionTreeClassifier__max_leaf_nodes': [2,3, None],
                # 'DecisionTreeClassifier__max_depth': [2,3, None],
                # 'DecisionTreeClassifier__min_samples_leaf': [1,3,5, None]
            }
        elif  method == 'RandomizedSearchCV':
            pass
        else:
            pass
        return parameters

    def svc_paramR(self,method='GridSearchCV'):
        if method == 'GridSearchCV':
            parameters = {
                'selector__pca__svd_solver': ['full', 'arpack', 'randomized'],
                'selector__pca__whiten': [True,False],
                'SVR__kernel': ['linear','poly', 'rbf','sigmoid'],
                # 'SVC__kernel': ['rbf'],
                'SVR__C': [1, 10, 100],

            }
        elif  method == 'RandomizedSearchCV':
            pass
        else:
            pass
        return parameters

    def mlperceptron_paramR(self,method='GridSearchCV'):
        if method == 'GridSearchCV':
            parameters = {
                'selector__pca__svd_solver': ['full', 'arpack', 'randomized'],
                'selector__pca__whiten': [True,False],
                'MLPRegressor__hidden_layer_sizes': [100],
                'MLPRegressor__activation': ['identity', 'logistic', 'tanh', 'relu']
            }
        elif  method == 'RandomizedSearchCV':
            pass
        else:
            pass
        return parameters

    ############################################################################

    def get_params(self, model, method):
        if self.problem_type == 'Classification':
            if model == 'AdaBoostClassifier':
                return self.adaboost_paramC(method)
            elif model == 'VotingClassifier':
                return self.voting_paramC(method)
            elif model == 'GradientBoostingClassifier':
                return self.gradientboosting_paramC(method)
            elif model == 'ExtraTreesClassifier':
                return self.extratrees_paramC(method)
            elif model == 'RandomForestClassifier':
                return self.randomforest_paramC(method)
            elif model == 'DecisionTreeClassifier':
                return self.decisiontree_paramC(method)
            elif model == 'LinearDiscriminantAnalysis':
                return self.lda_paramC(method)
            elif model == 'SVC':
                return self.svc_paramC(method)
            elif model == 'KNeighborsClassifier':
                return self.knn_paramC(method)
            elif model == 'LogisticRegression':
                return self.logistic_paramC(method)
            elif model == 'GaussianNB':
                return self.naivebayes_paramC(method)
            elif model == 'MLPClassifier':
                return self.mlperceptron_paramC(method)
        elif self.problem_type=='Regression':
            if model == 'KNeighborsRegressor':
                return self.knn_paramR(method)
            elif model == 'RandomForestRegressor':
                return self.randomforest_paramR(method)
            elif model == 'AdaBoostRegressor':
                return self.adaboost_paramR(method)
            elif model == 'BaggingRegressor':
                return self.bagging_paramR(method)
            elif model == 'ExtraTreesRegressor':
                return self.extratrees_paramR(method)
            elif model == 'GradientBoostingRegressor':
                return self.gradientboosting_paramR(method)
            elif model == 'DecisionTreeRegressor':
                return self.decisiontree_paramR(method)
            elif model == 'SVR':
                return self.svc_paramR(method)
            elif model == 'MLPRegressor':
                return self.mlperceptron_paramR(method)

        return None

    def evaluate_model(self, pipelines):
        n,m = pipelines
        parameters = self.get_params(n, self.optimizer)
        if self.optimizer == 'GridSearchCV':
            grid_search_t = GridSearchCV(m, parameters, verbose=1, cv=self.cv)
            print("Performing search...", n)
            grid_search_t.fit(self.evaluator.X_train, self.evaluator.y_train)
            cv_results = [round(x[1],5) for x in grid_search_t.grid_scores_]
            print(grid_search_t.best_params_)
            return cv_results
        elif self.optimizer == 'RandomizedSearchCV':
            random_search_t = RandomizedSearchCV(m, parameters, n_jobs=-1, verbose=1)
            print("Performing search...", n)
            random_search_t.fit(self.evaluator.X_train, self.evaluator.y_train)
            cv_results = [round(x[1],5) for x in random_search_t.grid_scores_]
            return cv_results
        else:
            pass

    def improve_grid_search(self):
        self.evaluator.split_data()
        self.report = [["Model", "Best_score"]]
        results = []

        num_cores=mp.cpu_count()
        print("****************** num_cores: ",num_cores," *********************")
        pool = mp.Pool(processes=num_cores)
        r = pool.map(self.evaluate_model,self.pipelines)
        pool.close()
        i=0

        for cv_results in r:
            print("Modeling ...", self.modelos[i])
            best_score_ = max(cv_results)
            d = {'name': self.modelos[i], 'values': cv_results, 'best_score': round(best_score_, 5)}
            results.append(d)
            self.report.append([self.modelos[i], round(best_score_, 5)])
            print("Best score: %0.3f" % best_score_)
            i = i+1
        print("*****************************************************************")

        self.score_report = sorted(results, key=lambda k: k['best_score'], reverse=True)
        headers = self.report.pop(0)
        df_report = pd.DataFrame(self.report, columns=headers)
        print(df_report)
        self.sort_report(df_report)


    def sort_report(self, report):
        """" Choose the best two algorithms"""
        report.sort_values(['Best_score'], ascending=[False], inplace=True)
        self.report = report.copy()
        #print(self.report)

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
        results = self.score_report
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
                y=[d['best_score'] for d in results],
                name='best score',
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
                    y=results[0]['best_score'],
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
                    y=results[-1]['best_score'],
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

        return self.plot_to_html(fig)

    def save_plot(self, path):
        with open(path, "w") as plot:
            plot.write(self.plot_html)

    def save_full_report(self, path):

        for index, elem in enumerate(self.full_report):
            elem.to_csv(path+'_model'+str(index+1)+'.csv', index=False)

    def save_score_report(self, path):

        self.score_report.to_csv(path+'_score'+'.csv', index=False)
